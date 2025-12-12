import os
import shutil

base_dir = r"c:\Repositories\seu-injection-framework\examples\flood_training_study"

moves = {
    "paper_markdown": [
        "01_introduction.md", "02_literature_review.md", "03_methodology.md", 
        "04_results.md", "05_discussion.md", "06_conclusion.md", "references.md"
    ],
    "reviews": [
        "ICLR_REVIEW_RESPONSE.md", "ICLR_REVIEW_V2_RESPONSE.md"
    ],
    "audit": [
        "CHANGES.md", "content_audit.md", "PAPER_READY.md"
    ],
    "data": [
        "comprehensive_results.csv", "comprehensive_results.json"
    ]
}

for folder, files in moves.items():
    target_dir = os.path.join(base_dir, folder)
    for file in files:
        src = os.path.join(base_dir, file)
        dst = os.path.join(target_dir, file)
        if os.path.exists(src):
            print(f"Moving {src} to {dst}")
            shutil.move(src, dst)
        else:
            print(f"File not found: {src}")
