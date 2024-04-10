"""
A custom `DataSet` for creating TFRecords files.
"""

import random
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional
from functools import partial

import tensorflow as tf
from kedro.io import AbstractVersionedDataSet, Version
from loguru import logger


class TfRecordsDataSet(AbstractVersionedDataSet):
    """
    A custom `DataSet` for creating TFRecords files.
    """

    def __init__(
        self,
        filepath: PurePosixPath,
        version: Optional[Version] = None,
        verbose: bool = True,
        compressed: bool = False,
    ):
        """
        Args:
            filepath: The path to the output TFRecords file.
            version: The version information for the `DataSet`.
            verbose: Whether to provide logging output every time a dataset
                is saved or loaded.
            compressed: If true, the saved files will be compressed.
        """
        super().__init__(PurePosixPath(filepath), version)

        self.__verbose = verbose
        self.__compressed = compressed

    def _load(self) -> tf.data.TFRecordDataset:
        """
        Loads the TFRecord file.

        Returns:
            The raw `tf.data.TFRecordDataset` that it loaded.

        """
        load_path = Path(self._get_load_path())
        if self.__verbose:
            logger.debug("Loading TFRecords from {}.", load_path)

        load_paths = [load_path.as_posix()]
        if load_path.is_dir():
            # Load everything in the directory as a single dataset.
            tfrecords_files = list(load_path.iterdir())
            logger.debug(
                "Loading {} TFRecords files from directory.",
                len(tfrecords_files),
            )
            # Shuffle them so the order appears random.
            random.shuffle(tfrecords_files)

            load_paths = [f.as_posix() for f in tfrecords_files]

        raw_dataset = tf.data.TFRecordDataset(load_paths)

        return raw_dataset

    def _save(self, examples: Iterable[tf.train.Example]) -> None:
        """
        Saves data to a TFRecord file.

        Args:
            examples: Examples to write will be pulled from this iterable and
                saved in-order.

        """
        save_path = self._get_save_path()
        if self.__verbose:
            logger.debug("Saving TFRecords to {}.", save_path)

        # Make sure all the intermediate directories are created.
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        create_writer = partial(tf.io.TFRecordWriter, save_path.as_posix())
        if self.__compressed:
            options = tf.io.TFRecordOptions(compression_type="GZIP")
            create_writer = partial(create_writer, options=options)
        with create_writer() as writer:
            for example in examples:
                serialized = example.SerializeToString()
                writer.write(serialized)

        logger.debug("Done saving data.")

    def _exists(self) -> bool:
        path = self._get_load_path()
        return Path(path.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(version=self._version)
