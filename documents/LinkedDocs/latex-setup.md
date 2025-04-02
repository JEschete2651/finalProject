# 📄 Setting Up a LaTeX Environment

For any documentation or report writing that requires LaTeX, we recommend using **TeXstudio** in combination with **MiKTeX**. This setup provides a complete LaTeX development environment on Windows.

---

## 🧰 Required Tools

- **TeXstudio** (LaTeX editor):  
  [https://www.texstudio.org/](https://www.texstudio.org/)

- **MiKTeX** (LaTeX distribution):  
  [https://miktex.org/](https://miktex.org/)

---

## 🛠️ Setup Instructions

1. **Install MiKTeX**  
   - Download and install MiKTeX from the link above.
   - Open the **MiKTeX Console** after installation and allow it to install missing packages on-the-fly if prompted.
   - Use the *Updates* tab to stay current.

2. **Install TeXstudio**  
   - Download and install TeXstudio from its official site.
   - It should auto-detect MiKTeX. If not, configure it in `Options > Configure TeXstudio > Commands`.

3. **Compile a `.tex` File**  
   - Open or create a `.tex` file in TeXstudio.
   - Use `Ctrl + Shift + B` or the green *Build & View* button to generate a PDF.

---

## 📚 LaTeX Reference

- Full language guide and searchable reference:  
  [https://devdocs.io/latex/](https://devdocs.io/latex/)

> Note: If you're contributing `.tex` documents to this repo, commit your changes under a branch prefixed with `doc/<topic>`.
