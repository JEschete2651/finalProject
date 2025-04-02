# Setting Up a LaTeX Environment

For writing documentation, reports, or technical papers in LaTeX, we recommend using TeXstudio as the editor and MiKTeX as the LaTeX distribution. This setup provides full offline compilation and flexibility for advanced formatting, such as conference papers.

## Required Tools

- TeXstudio (LaTeX editor)  
  https://www.texstudio.org/

- MiKTeX (LaTeX distribution)  
  https://miktex.org/

## Setup Instructions

1. Install MiKTeX  
   - Download from the link above and complete the installation.
   - Open the MiKTeX Console after installation and enable the option to install missing packages on-the-fly.
   - Keep the distribution up to date using the Updates tab in the MiKTeX Console.

2. Install TeXstudio  
   - Download and install from the official site.
   - TeXstudio should automatically detect MiKTeX. If not, you can manually configure it under:
     `Options > Configure TeXstudio > Commands`

3. Compile a LaTeX file  
   - Open or create a `.tex` file in TeXstudio.
   - Use `Ctrl + Shift + B` or the green "Build & View" button to compile the document and generate a PDF.

## LaTeX Learning Resources

- LaTeX Language Reference  
  https://devdocs.io/latex/

- Udemy Course: *LaTeX for Professional Publications*  
  https://www.udemy.com/share/101GoS3@A1AAkUfSP5p1h48OhpmCEZc__zKLvNl828LddD-UD_M4WDYS4lA5XSZkGQsLiuUIzQ==/  
  This course provides a complete introduction to LaTeX, covering formatting, tables, figures, citations, and more. Recommended for anyone writing academic or technical documents.

## Contribution Notes

If you are contributing LaTeX documents to this project, commit them under a branch named using the format `doc/<topic>` and place them inside the `documents/` directory.
