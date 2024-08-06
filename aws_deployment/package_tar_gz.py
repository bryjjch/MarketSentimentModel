import tarfile

def create_model_tar_gz(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=".")

create_model_tar_gz("model.tar.gz", "my_model")
