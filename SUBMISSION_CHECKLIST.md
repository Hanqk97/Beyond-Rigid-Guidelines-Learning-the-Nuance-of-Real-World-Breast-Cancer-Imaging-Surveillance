# Submission Checklist

Before creating the ML4H supplementary ZIP:

- Confirm the venue permits code plus aggregate outputs without the licensed data.
- Confirm each CSV in `aggregate_results/` is permitted by the data-use agreement.
- Confirm no author name, affiliation, email, username, repository URL, or local absolute path remains during anonymous review.
- Keep the remote repository private and do not provide its identifying URL during anonymous review.
- Confirm there are no raw/prepared data files, patient/event rows, prediction-detail files, bootstrap caches, checkpoints, logs, notebooks with embedded outputs, or hidden environment files.
- Open both generated plot PDFs and compare them with the manuscript.
- Copy this repository, remove the copied `.git/` directory, and create the ZIP from the copied repository contents, not from its parent research workspace.
- Inspect the ZIP file listing once more before uploading.

For the camera-ready GitHub repository:

- Replace “Anonymous Authors” in `LICENSE`.
- Add the final paper citation, DOI/link, author/contact metadata, and repository URL.
- Reconfirm the license and data-availability language with the data-use agreement.
- Add a tagged release matching the camera-ready artifact.
