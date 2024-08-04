import os
import glob
from datetime import datetime

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Documentation</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }}
        header {{
            background: #333;
            color: #fff;
            padding: 20px 0;
            text-align: center;
            text-transform: uppercase;
            font-size: 1.5em;
        }}
        .container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .content {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 800px;
            text-align: center;
        }}
        .content h2 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .content p {{
            margin: 10px 0;
            font-size: 1.2em;
        }}
        .content a {{
            color: #007bff;
            text-decoration: none;
        }}
        .content a:hover {{
            text-decoration: underline;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            background: #333;
            color: #fff;
        }}
    </style>
</head>
<body>
    <header>
        Welcome to the Project Documentation
    </header>
    <div class="container">
        <div class="content">
            <h2>Latest Documentation</h2>
            <p><a href="docs/html/index.html">{documentation_name}</a></p>
            <h2>Previous Reports</h2>
            {links}
        </div>
    </div>
    <footer>
        &copy; 2024 Project Documentation
    </footer>
</body>
</html>
'''

if __name__ == "__main__":
    report_files = glob.glob('reports/**/report.html', recursive=True)
    links_list = []
    for report in sorted(report_files):
        rel_path = os.path.relpath(report, ".")
        links_list.append(f'<p><a href="{rel_path}">{report.split("/")[-2]}</a></p>')
    links = ''.join(links_list)

    html_content = html_template.format(links = links,
                                        documentation_name = f'Generated {datetime.now()}')
    print(html_content)
