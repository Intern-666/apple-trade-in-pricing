import os
import re

def patch_file(filepath, replacements, regex_replacements=None):
    if not os.path.exists(filepath):
        print(f"Skipped: {filepath} not found.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Apply precise string replacements
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"Warning: Expected string not found in {filepath}:\n{old.strip()}")

    # Apply regex replacements
    if regex_replacements:
        for pattern, new in regex_replacements:
            new_content = re.sub(pattern, new, content, flags=re.DOTALL)
            if new_content == content:
                print(f"Warning: Regex pattern matched nothing in {filepath}")
            content = new_content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Patched: {filepath}")

def main():
    # 1. Patch server/app.py
    app_replacements = [
        (', "Chipset"', ''),
        ('    Chipset: Optional[str] = None\n', ''),
        ('    chipset = item.Chipset.strip() if item.Chipset else np.nan\n\n    if device in ("iPhone", "iPad", "Mac") and pd.isna(chipset):\n\n        raise HTTPException(status_code=400, detail="Chipset is required.")\n\n', ''),
        (' "Chipset": chipset,', ''),
        ('                    "chipset": (None if pd.isna(row["Chipset"]) else str(row["Chipset"]).strip()),\n', ''),
        (' "chipset": clean_value("Chipset", row),', '')
    ]
    patch_file("server/app.py", app_replacements)

    # 2. Patch internal/bulk_import.py
    bulk_replacements = [
        ('    "Chipset",\n', ''),
        ('    "Chipset": ["chipset", "chip", "processor", "soc"],\n', ''),
        ('    if field == "Chipset":\n        return False\n\n', ''),
        ('        "chipset": normalize_text(_get_mapped_value(row, "Chipset", mapped_fields)) or UNKNOWN,\n', ''),
        (' "Chipset": "chipset",', ''),
        ('def validate_new_row_fields(device, sub_device, model_name, msrp, trade_in_value, storage, storage_type, chipset, connectivity, material, case_size, charging_method, model_year):', 'def validate_new_row_fields(device, sub_device, model_name, msrp, trade_in_value, storage, storage_type, connectivity, material, case_size, charging_method, model_year):')
    ]
    bulk_regex = [
        (r'def chipset_similarity\(.*?return difflib\.SequenceMatcher\(None, i_text, m_text\)\.ratio\(\)\n', '')
    ]
    patch_file("internal/bulk_import.py", bulk_replacements, bulk_regex)

    # 3. Patch assets/admin.html
    admin_html_replacements = [
        ('\t\t\t\tdocument.getElementById("add-chipset-group").classList.toggle("hidden", !["iPhone", "iPad", "Mac"].includes(device));\n', ''),
        ('\t\t\t\tdocument.getElementById("add-chipset").value = "";\n', ''),
        ('\t\t\tconst chipset = document.getElementById("add-chipset").value.trim();\n', ''),
        ('\t\t\t\tChipset: chipset || null,\n', '')
    ]
    admin_html_regex = [
        (r'\t\t\t\t<div class="form-group" id="add-chipset-group">.*?</div>\n', ''),
        (r'\t\t\tif \(\n\t\t\t\t\["iPhone", "iPad", "Mac"\]\.includes\(device\) && !chipset\) \{\n\t\t\t\tshowAdminToast\("Please enter the chipset\.", "warning"\);\n\t\t\t\treturn;\n\t\t\t\}\n', '')
    ]
    patch_file("assets/admin.html", admin_html_replacements, admin_html_regex)

    # 4. Patch internal/test_bulk_import.py
    test_replacements = [
        ('    chipset="A13 Bionic",\n', ''),
        ('        "Chipset": chipset,\n', ''),
        ('"different_fields": ["Max. Trade-In Value (RM)", "Chipset"]', '"different_fields": ["Max. Trade-In Value (RM)", "Storage Type"]')
    ]
    patch_file("internal/test_bulk_import.py", test_replacements)

    print("\nPatching complete. Run tests to verify:")
    print("python -m unittest discover -s internal -p 'test_*.py' -v")

if __name__ == "__main__":
    main()