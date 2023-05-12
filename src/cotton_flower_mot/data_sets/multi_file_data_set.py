"""
This is actually a sort of meta-dataset that is designed to wrap other
datasets which produce a file. It take an arbitrary number of the outputs
from one of these datasets and save them to (or load them from) a single
directory.
"""


from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional

from kedro.io import AbstractDataSet, AbstractVersionedDataSet, Version
from loguru import logger


class MultiFileDataSet(AbstractVersionedDataSet):
    """
    This is actually a sort of meta-dataset that is designed to wrap other
    datasets which produce a file. It take an arbitrary number of the outputs
    from one of these datasets and save them to (or load them from) a single
    directory.
    """

    def __init__(
        self,
        filepath: PurePosixPath,
        dataset: str,
        extension: str,
        file_name_format: str = "data_{index}",
        version: Optional[Version] = None,
        skip_missing: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            filepath: The path to the output directory.
            dataset: The name of the dataset module that we are wrapping.
            extension: Extension to use for created files.
            file_name_format: The format to use for the individual files in
                the dataset. The variable "index" can be used in the format, and
                will be an integer that indicates the specific file number.
            version: The version information for the `DataSet`.
            skip_missing: If true, it will simply skip missing data files
                when loading instead of failing. The loading iterator will
                return `None` for skipped files.
            **kwargs: Will be forwarded to the internal datasets.
        """
        super().__init__(PurePosixPath(filepath), version)

        self.__dataset_type = dataset
        self.__version = version
        self.__dataset_config = kwargs
        self.__extension = extension
        self.__file_name_format = file_name_format
        self.__skip_missing = skip_missing

        # The internal datasets that we use for saving and loading.
        self.__datasets = []
        # Default base path to use.
        self.__default_base_path = self._get_save_path()

    def __create_dataset(
        self, index: int, *, base_path: Path
    ) -> AbstractDataSet:
        """
        Creates a new dataset.

        Args:
            index: The index to use in the dataset's name.
            base_path: The location of the directory where the data in this
                dataset is stored.

        Returns:
            The dataset that it created.

        """
        # Construct the file path.
        file_name = self.__file_name_format.format(index=index)
        filepath = base_path / f"{file_name}{self.__extension}"
        logger.debug(
            "Creating {} to save data at {}.", self.__dataset_type, filepath
        )
        config = dict(
            filepath=filepath.as_posix(),
            type=self.__dataset_type,
            **self.__dataset_config,
        )

        load_version = None
        save_version = None
        if self.__version is not None:
            # Specify load and save versions.
            load_version = self.__version.load
            save_version = self.__version.save

        return AbstractVersionedDataSet.from_config(
            f"{self.__dataset_type}_{index}",
            config,
            load_version=load_version,
            save_version=save_version,
        )

    def __iter_datasets(
        self, base_path: Optional[Path] = None
    ) -> Iterable[AbstractDataSet]:
        """
        Iterates through all of the internal datasets in-order, creating
        those that don't exist on-the-fly.

        Args:
            base_path: The base path to use when creating datasets. Defaults
                to the save path.

        Yields:
              The internal datasets.

        """
        yield from self.__datasets

        if base_path is None:
            base_path = self.__default_base_path

        # At this point, we need to start adding new ones.
        while True:
            dataset = self.__create_dataset(
                len(self.__datasets), base_path=base_path
            )
            logger.debug("Created dataset at {}.", base_path)
            self.__datasets.append(dataset)

            yield dataset

    def _load(self) -> Iterable[Any]:
        # Figure out how many saved files there are.
        load_path = Path(self._get_load_path())
        saved_files = load_path.iterdir()

        for _, dataset in zip(
            saved_files, self.__iter_datasets(base_path=load_path)
        ):
            if not dataset.exists() and self.__skip_missing:
                # Skip the missing dataset.
                logger.info("Skipping non-existent dataset load.")
                yield None

            else:
                yield dataset.load()

    def _save(self, data: Iterable[Any]) -> None:
        for data_item, dataset in zip(data, self.__iter_datasets()):
            # Save the data.
            dataset.save(data_item)

    def _exists(self) -> bool:
        path = self._get_load_path()
        return Path(path).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            dataset=self.__dataset_type,
            version=self.__version,
            extension=self.__extension,
            config=self.__dataset_config,
        )
