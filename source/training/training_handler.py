# training/training_handler.py

import logging
import io
import matplotlib.pyplot as plt
from  tensorflow.keras.callbacks import Callback
from typing import Optional
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

from .training_config import TrainingConfig
from ..environment.trading_environment import TradingEnvironment
from ..agent.agent_handler import AgentHandler
from ..plotting.plot_responsibility_chain_base import PlotResponsibilityChainBase
from ..plotting.plot_testing_history_responsibility_chain import PlotTestingHistoryResponsibilityChain
from ..plotting.plot_training_history_responsibility_chain import PlotTrainingHistoryResponsibilityChain

class TrainingHandler():
    """
    Responsible for orchestrating the training process and report generation.

    This class manages the complete training workflow, from initializing the
    environment and agent, running training and testing sessions, to generating
    PDF reports with performance visualizations and logs. It serves as the main
    entry point for executing and documenting trading agent training.
    """

    """
    PDF report related constants
    """
    HEADING_SPACING = 20
    CAPTION_FONT_SIZE = 14
    TEXT_FONT_SIZE = 8
    FONT_NAME = 'Courier'
    MARGINS = {
        'left': 30,
        'right': 30,
        'top': 30,
        'bottom': 30
    }
    EXCLUDE_FROM_LOGS = ['ETA']

    def __init__(self, config: TrainingConfig, page_width: int = letter[0], page_height: int = letter[1],
                 heading_spacing: int = HEADING_SPACING, caption_font_size: int = CAPTION_FONT_SIZE,
                 text_font_size: int = TEXT_FONT_SIZE, font_name: str = FONT_NAME,
                 margins: dict[str, int] = MARGINS, exclude_from_logs: list[str] = EXCLUDE_FROM_LOGS) -> None:
        """
        Initializes the training handler with configuration parameters.

        Parameters:
            config (TrainingConfig): Configuration containing environment and agent settings.
            page_width (int): Width of PDF report pages in points.
            page_height (int): Height of PDF report pages in points.
            heading_spacing (int): Spacing between headings and content in points.
            caption_font_size (int): Font size for section captions.
            text_font_size (int): Font size for main text content.
            font_name (str): Font family to use for text in reports.
            margins (dict[str, int]): Dictionary with 'left', 'right', 'top', and 'bottom' margins.
            exclude_from_logs (list[str]): Terms that should be excluded from logs in reports.

        Raises:
            ValueError: If margins dictionary doesn't contain required keys.
        """

        # Training related configuration
        self.__environment: TradingEnvironment = config.instantiate_environment()
        self.__agent: AgentHandler = config.instantiate_agent()
        self.__nr_of_steps: int = config.nr_of_steps
        self.__repeat_test: int = config.repeat_test
        self.__steps_per_episode: int = int(config.nr_of_steps / config.nr_of_episodes)

        # Report related configuration
        self.__config_summary = str(config)
        self.__plotting_chain: PlotResponsibilityChainBase = PlotTestingHistoryResponsibilityChain()
        self.__plotting_chain.add_next_chain_link(PlotTrainingHistoryResponsibilityChain())
        self.__generated_data: dict = {}
        self.__logs: io.StringIO = io.StringIO()
        self.__page_width = page_width
        self.__page_height = page_height
        self.__heading_spacing = heading_spacing
        self.__caption_font_size = caption_font_size
        self.__text_font_size = text_font_size
        self.__font_name = font_name
        self.__margins = margins
        self.__exclude_from_logs = exclude_from_logs

        if not any(term in margins for term in ['left', 'right', 'top', 'bottom']):
            raise ValueError("Margins should contain 'left', 'right', 'top' and 'bottom' keys!")

    def run_training(self, callbacks: list[Callback] = [], weights_load_path: Optional[str] = None,
                     weights_save_path: Optional[str] = None) -> None:
        """
        Executes the training and testing process for the trading agent.

        This method orchestrates the complete training workflow, capturing logs,
        training the agent, and testing its performance. It populates internal
        data structures with results that can later be used for report generation.

        Parameters:
            callbacks (list[Callback]): Keras callbacks to use during training.
            weights_load_path (str, optional): Path to load pre-trained weights from.
            weights_save_path (str, optional): Path to save trained weights to.
        """

        log_streamer = logging.StreamHandler(self.__logs)
        log_streamer.setFormatter(logging.Formatter('%(message)s'))
        log_streamer.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_streamer)

        try:
            logging.info(f"Training started!")
            logging.info(self.__config_summary.replace('\t', '  '))
            logging.info(f"Printing models architecture...")
            self.__agent.print_model_summary(print_function = lambda x: logging.info(x))

            self.__generated_data['train'] = self.__agent.train_agent(self.__environment,
                                                                    self.__nr_of_steps,
                                                                    self.__steps_per_episode,
                                                                    callbacks,
                                                                    weights_load_path,
                                                                    weights_save_path)
            self.__generated_data['test'] = self.__agent.test_agent(self.__environment,
                                                                    self.__repeat_test)

            logging.info(f"Training finished!")
        except Exception as e:
            logging.error(f"Training failed! Original error: {e}")
        finally:
            root_logger.removeHandler(log_streamer)
            log_streamer.close()

    def __handle_plot_generation(self, data: dict) -> Optional[ImageReader]:
        """
        Generates a plot based on provided data using the responsibility chain.

        Parameters:
            data (dict): Dictionary containing 'key' identifying the plot type
                        and 'plot_data' containing the actual data to be plotted.

        Returns:
            Optional[ImageReader]: ReportLab ImageReader object if plot was generated,
                                 None if no handler could process the request.
        """

        image_reader = None
        axes = self.__plotting_chain.plot(data)
        if axes is not None:
            buffer = io.BytesIO()
            axes.figure.savefig(buffer, format = 'png')
            buffer.seek(0)
            image_reader = ImageReader(buffer)
            plt.close(axes.figure)
        else:
            logging.warning(f'Did not managed to generate plot for {data["key"]}!')

        return image_reader

    def __calculate_max_dimensions(self, pdf: canvas.Canvas) -> tuple[int, int]:
        """
        Calculates maximum text dimensions that can fit on a PDF page.

        Parameters:
            pdf (canvas.Canvas): The PDF canvas to calculate dimensions for.

        Returns:
            tuple[int, int]: Maximum number of characters per line and number
                           of lines per page that can fit within margins.
        """

        pdf.setFont(self.__font_name, self.__text_font_size)
        text_width = pdf.stringWidth(' ', self.__font_name, self.__text_font_size)
        available_width = self.__page_width - self.__margins['left'] - self.__margins['right']
        max_width = int(available_width / text_width)

        text_height = self.__text_font_size * 1.2
        available_height = self.__page_height - self.__margins['top'] - self.__margins['bottom'] \
            - self.__heading_spacing
        max_height = int(available_height / text_height)

        return max_width, max_height

    def __handle_logs_preprocessing(self, raw_logs_bufffer: list[str], max_log_length: int,
                                    max_lines_per_page: int) -> list[list[str]]:
        """
        Processes raw logs to fit them within PDF page constraints.

        This method filters out excluded terms, handles line wrapping for long lines,
        and chunks the logs into page-sized portions.

        Parameters:
            raw_logs_bufffer (list[str]): Raw log lines to process.
            max_log_length (int): Maximum characters per line.
            max_lines_per_page (int): Maximum lines per page.

        Returns:
            list[list[str]]: List of pages, where each page is a list of log lines.
        """

        preprocessed_logs = []
        for log in raw_logs_bufffer:
            if not any(exclude_term in log for exclude_term in self.__exclude_from_logs):
                if len(log) > max_log_length:
                    for i in range(0, len(log), max_log_length):
                        preprocessed_logs.append(log[i:i + max_log_length])
                else:
                    preprocessed_logs.append(log)

        return [preprocessed_logs[i:i + max_lines_per_page]
                for i in range(0, len(preprocessed_logs), max_lines_per_page)]

    def __draw_caption(self, pdf: canvas.Canvas, text: str) -> None:
        """
        Draws a section caption with separating line on the PDF.

        Parameters:
            pdf (canvas.Canvas): PDF canvas to draw on.
            text (str): Caption text to draw.
        """

        pdf.setFont(self.__font_name, self.__caption_font_size)
        pdf.drawString(self.__margins['left'], self.__page_height - self.__margins['top'], text)
        pdf.setLineWidth(2)
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.line(self.__margins['left'], self.__page_height - self.__margins['top'] \
                 - self.__heading_spacing / 2, self.__page_width - self.__margins['right'],
                 self.__page_height - self.__margins['top'] - self.__heading_spacing / 2)

    def __draw_text_body(self, pdf: canvas.Canvas, text_body: list[str]) -> None:
        """
        Draws a block of text lines on the PDF.

        Parameters:
            pdf (canvas.Canvas): PDF canvas to draw on.
            text_body (list[str]): List of text lines to draw.
        """

        text_block = pdf.beginText(self.__margins['left'], self.__page_height - \
            self.__margins['top'] - self.__heading_spacing)
        text_block.setFont(self.__font_name, self.__text_font_size)
        for line in text_body:
            text_block.textLine(line)
        pdf.drawText(text_block)

    def generate_report(self, path_to_pdf: str) -> None:
        """
        Generates a comprehensive PDF report of training and testing results.

        Creates a multi-page report with logs, training history plots, and test
        results visualizations based on the data collected during training.

        Parameters:
            path_to_pdf (str): File path where the PDF report should be saved.
        """

        logging.info(f"Generating report...")
        pdf = canvas.Canvas(path_to_pdf, pagesize = letter)
        pdf.setTitle("Report")

        # Report logs
        raw_logs = self.__logs.getvalue().split('\n')
        max_log_length, max_lines_per_page = self.__calculate_max_dimensions(pdf)
        preprocessed_logs = self.__handle_logs_preprocessing(raw_logs, max_log_length, max_lines_per_page)
        for preprocessed_logs_chunk in preprocessed_logs:
            self.__draw_caption(pdf, "Log output")
            self.__draw_text_body(pdf, preprocessed_logs_chunk)
            pdf.showPage()

        # Draw training plot
        data = {
            'key': 'training_history',
            'plot_data': self.__generated_data['train']
        }
        plot_buffer = self.__handle_plot_generation(data)
        if plot_buffer is not None:
            self.__draw_caption(pdf, "Training performance")
            pdf.drawImage(plot_buffer, inch, self.__page_height - 7.5 * inch, width = 6 * inch,
                          preserveAspectRatio = True)
            pdf.showPage()

        # Draw testing plots
        for index, testing_data in self.__generated_data['test'].items():
            data = {
                'key': 'testing_history',
                'plot_data': testing_data
            }
            plot_buffer = self.__handle_plot_generation(data)
            if plot_buffer is not None:
                self.__draw_caption(pdf, f"Testing outcome, trial: {index + 1}")
                pdf.drawImage(plot_buffer, inch, self.__page_height - 7.5 * inch, width = 6 * inch,
                              preserveAspectRatio = True)
                pdf.showPage()

        pdf.save()
        logging.info(f"Report generated!")
