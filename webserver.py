from flask import Flask, render_template, send_from_directory, jsonify
import os
import glob
from datetime import datetime
import json

app = Flask(__name__)

# Create templates and static directories if they don't exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)


def get_html_files():
    """Scan the output directory for HTML files and return their metadata"""
    html_files = []
    output_dir = 'output'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in glob.glob(os.path.join(output_dir, '*.html')):
        try:
            filename = os.path.basename(file)
            # timestamp = filename.split('_')[1].split('.')[0]
            # date = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')

            with open(file) as f:
                content = f.read()
                title = content.split('<title>')[1].split('</title>')[0] if '<title>' in content else 'Untitled'

            html_files.append({
                'filename': filename,
                'filepath': file,
                # 'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                # 'timestamp': int(date.timestamp() * 1000),
                'title': title
            })
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return sorted(html_files, key=lambda x: x['filepath'], reverse=True)


@app.route('/')
def index():
    files = get_html_files()
    return render_template('index.html', files=files)


@app.route('/content/<filename>')
def get_content(filename):
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    file_path = os.path.join(output_dir, filename)

    if not os.path.exists(file_path):
        return {'success': False, 'error': 'File not found'}, 404

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {'success': True, 'content': content}
    except Exception as e:
        return {'success': False, 'error': str(e)}, 500


@app.route('/view/<path:filename>')
def view_file(filename):
    return send_from_directory('output', filename)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
