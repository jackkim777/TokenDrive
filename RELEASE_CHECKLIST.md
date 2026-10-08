# TokenDrive Release Checklist

1. Create a private GitHub repository named `TokenDrive`.
2. Push this `JCDE` folder as the repository root.
3. Upload checkpoints, ONNX exports, and large data artifacts to Hugging Face or another large-file host.
4. Replace the placeholder URLs in `README.md` with the final GitHub and Hugging Face URLs.
5. Clone the repository into a fresh folder and run the README installation, label-generation, training smoke, and evaluation commands.
6. Tag the paper-matched release, for example `v1.0-jcde`.
7. Switch the GitHub repository from Private to Public after the release check passes.
8. Add the final public code/model/data links to the paper's Data Availability section.
