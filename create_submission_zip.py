import os
import zipfile

def create_submission_zip(source_dir, output_zip):
    exclude_dirs = {
        'venv',
        '.git',
        '.gemini',
        'catboost_info',
        '__pycache__',
        'scratch',
        '.system_generated'
    }
    exclude_files = {
        'train.json',
        'test.json',
        'create_submission_zip.py',
        'SUMMARY.md'
    }
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if file in exclude_files:
                    continue
                # Also exclude any file that looks like a dataset CSV
                if 'train' in file.lower() and file.lower().endswith('.csv'):
                    continue
                if 'test' in file.lower() and file.lower().endswith('.csv'):
                    continue
                # Exclude python bytecode
                if file.endswith('.pyc') or file.endswith('.pyo'):
                    continue
                # Exclude report artifacts
                if file.endswith('.pdf') or file.endswith('.tex') or file.endswith('.aux'):
                    continue
                if file == 'report.log':
                    continue
                
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, source_dir)
                zipf.write(filepath, arcname)
                print(f"Added: {arcname}")

if __name__ == "__main__":
    source = r"d:\IISC SEM1\ML\Project"
    output = r"d:\IISC SEM1\ML\Submission_FeatureForge.zip"
    create_submission_zip(source, output)
    print(f"\nCreated zip file: {output}")
