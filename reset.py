# Backward-compatibility wrapper for reset.py
if __name__ == "__main__":
    import runpy
    runpy.run_path("scripts/reset_db.py")
