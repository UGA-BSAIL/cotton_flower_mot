# coding: utf-8

"""
    CVAT REST API

    REST API for Computer Vision Annotation Tool (CVAT)  # noqa: E501

    OpenAPI spec version: v1
    Contact: nikita.manovich@intel.com
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class Project(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'url': 'str',
        'id': 'int',
        'name': 'str',
        'labels': 'list[Label]',
        'tasks': 'list[Task]',
        'owner': 'BasicUser',
        'assignee': 'BasicUser',
        'owner_id': 'int',
        'assignee_id': 'int',
        'bug_tracker': 'str',
        'task_subsets': 'list[str]',
        'created_date': 'datetime',
        'updated_date': 'datetime',
        'status': 'str',
        'training_project': 'TrainingProject',
        'dimension': 'str'
    }

    attribute_map = {
        'url': 'url',
        'id': 'id',
        'name': 'name',
        'labels': 'labels',
        'tasks': 'tasks',
        'owner': 'owner',
        'assignee': 'assignee',
        'owner_id': 'owner_id',
        'assignee_id': 'assignee_id',
        'bug_tracker': 'bug_tracker',
        'task_subsets': 'task_subsets',
        'created_date': 'created_date',
        'updated_date': 'updated_date',
        'status': 'status',
        'training_project': 'training_project',
        'dimension': 'dimension'
    }

    def __init__(self, url=None, id=None, name=None, labels=None, tasks=None, owner=None, assignee=None, owner_id=None, assignee_id=None, bug_tracker=None, task_subsets=None, created_date=None, updated_date=None, status=None, training_project=None, dimension=None):  # noqa: E501
        """Project - a model defined in Swagger"""  # noqa: E501
        self._url = None
        self._id = None
        self._name = None
        self._labels = None
        self._tasks = None
        self._owner = None
        self._assignee = None
        self._owner_id = None
        self._assignee_id = None
        self._bug_tracker = None
        self._task_subsets = None
        self._created_date = None
        self._updated_date = None
        self._status = None
        self._training_project = None
        self._dimension = None
        self.discriminator = None
        if url is not None:
            self.url = url
        if id is not None:
            self.id = id
        self.name = name
        if labels is not None:
            self.labels = labels
        if tasks is not None:
            self.tasks = tasks
        if owner is not None:
            self.owner = owner
        if assignee is not None:
            self.assignee = assignee
        if owner_id is not None:
            self.owner_id = owner_id
        if assignee_id is not None:
            self.assignee_id = assignee_id
        if bug_tracker is not None:
            self.bug_tracker = bug_tracker
        if task_subsets is not None:
            self.task_subsets = task_subsets
        if created_date is not None:
            self.created_date = created_date
        if updated_date is not None:
            self.updated_date = updated_date
        if status is not None:
            self.status = status
        if training_project is not None:
            self.training_project = training_project
        if dimension is not None:
            self.dimension = dimension

    @property
    def url(self):
        """Gets the url of this Project.  # noqa: E501


        :return: The url of this Project.  # noqa: E501
        :rtype: str
        """
        return self._url

    @url.setter
    def url(self, url):
        """Sets the url of this Project.


        :param url: The url of this Project.  # noqa: E501
        :type: str
        """

        self._url = url

    @property
    def id(self):
        """Gets the id of this Project.  # noqa: E501


        :return: The id of this Project.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this Project.


        :param id: The id of this Project.  # noqa: E501
        :type: int
        """

        self._id = id

    @property
    def name(self):
        """Gets the name of this Project.  # noqa: E501


        :return: The name of this Project.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this Project.


        :param name: The name of this Project.  # noqa: E501
        :type: str
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def labels(self):
        """Gets the labels of this Project.  # noqa: E501


        :return: The labels of this Project.  # noqa: E501
        :rtype: list[Label]
        """
        return self._labels

    @labels.setter
    def labels(self, labels):
        """Sets the labels of this Project.


        :param labels: The labels of this Project.  # noqa: E501
        :type: list[Label]
        """

        self._labels = labels

    @property
    def tasks(self):
        """Gets the tasks of this Project.  # noqa: E501


        :return: The tasks of this Project.  # noqa: E501
        :rtype: list[Task]
        """
        return self._tasks

    @tasks.setter
    def tasks(self, tasks):
        """Sets the tasks of this Project.


        :param tasks: The tasks of this Project.  # noqa: E501
        :type: list[Task]
        """

        self._tasks = tasks

    @property
    def owner(self):
        """Gets the owner of this Project.  # noqa: E501


        :return: The owner of this Project.  # noqa: E501
        :rtype: BasicUser
        """
        return self._owner

    @owner.setter
    def owner(self, owner):
        """Sets the owner of this Project.


        :param owner: The owner of this Project.  # noqa: E501
        :type: BasicUser
        """

        self._owner = owner

    @property
    def assignee(self):
        """Gets the assignee of this Project.  # noqa: E501


        :return: The assignee of this Project.  # noqa: E501
        :rtype: BasicUser
        """
        return self._assignee

    @assignee.setter
    def assignee(self, assignee):
        """Sets the assignee of this Project.


        :param assignee: The assignee of this Project.  # noqa: E501
        :type: BasicUser
        """

        self._assignee = assignee

    @property
    def owner_id(self):
        """Gets the owner_id of this Project.  # noqa: E501


        :return: The owner_id of this Project.  # noqa: E501
        :rtype: int
        """
        return self._owner_id

    @owner_id.setter
    def owner_id(self, owner_id):
        """Sets the owner_id of this Project.


        :param owner_id: The owner_id of this Project.  # noqa: E501
        :type: int
        """

        self._owner_id = owner_id

    @property
    def assignee_id(self):
        """Gets the assignee_id of this Project.  # noqa: E501


        :return: The assignee_id of this Project.  # noqa: E501
        :rtype: int
        """
        return self._assignee_id

    @assignee_id.setter
    def assignee_id(self, assignee_id):
        """Sets the assignee_id of this Project.


        :param assignee_id: The assignee_id of this Project.  # noqa: E501
        :type: int
        """

        self._assignee_id = assignee_id

    @property
    def bug_tracker(self):
        """Gets the bug_tracker of this Project.  # noqa: E501


        :return: The bug_tracker of this Project.  # noqa: E501
        :rtype: str
        """
        return self._bug_tracker

    @bug_tracker.setter
    def bug_tracker(self, bug_tracker):
        """Sets the bug_tracker of this Project.


        :param bug_tracker: The bug_tracker of this Project.  # noqa: E501
        :type: str
        """

        self._bug_tracker = bug_tracker

    @property
    def task_subsets(self):
        """Gets the task_subsets of this Project.  # noqa: E501


        :return: The task_subsets of this Project.  # noqa: E501
        :rtype: list[str]
        """
        return self._task_subsets

    @task_subsets.setter
    def task_subsets(self, task_subsets):
        """Sets the task_subsets of this Project.


        :param task_subsets: The task_subsets of this Project.  # noqa: E501
        :type: list[str]
        """

        self._task_subsets = task_subsets

    @property
    def created_date(self):
        """Gets the created_date of this Project.  # noqa: E501


        :return: The created_date of this Project.  # noqa: E501
        :rtype: datetime
        """
        return self._created_date

    @created_date.setter
    def created_date(self, created_date):
        """Sets the created_date of this Project.


        :param created_date: The created_date of this Project.  # noqa: E501
        :type: datetime
        """

        self._created_date = created_date

    @property
    def updated_date(self):
        """Gets the updated_date of this Project.  # noqa: E501


        :return: The updated_date of this Project.  # noqa: E501
        :rtype: datetime
        """
        return self._updated_date

    @updated_date.setter
    def updated_date(self, updated_date):
        """Sets the updated_date of this Project.


        :param updated_date: The updated_date of this Project.  # noqa: E501
        :type: datetime
        """

        self._updated_date = updated_date

    @property
    def status(self):
        """Gets the status of this Project.  # noqa: E501


        :return: The status of this Project.  # noqa: E501
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status):
        """Sets the status of this Project.


        :param status: The status of this Project.  # noqa: E501
        :type: str
        """
        allowed_values = ["annotation", "validation", "completed"]  # noqa: E501
        if status not in allowed_values:
            raise ValueError(
                "Invalid value for `status` ({0}), must be one of {1}"  # noqa: E501
                .format(status, allowed_values)
            )

        self._status = status

    @property
    def training_project(self):
        """Gets the training_project of this Project.  # noqa: E501


        :return: The training_project of this Project.  # noqa: E501
        :rtype: TrainingProject
        """
        return self._training_project

    @training_project.setter
    def training_project(self, training_project):
        """Sets the training_project of this Project.


        :param training_project: The training_project of this Project.  # noqa: E501
        :type: TrainingProject
        """

        self._training_project = training_project

    @property
    def dimension(self):
        """Gets the dimension of this Project.  # noqa: E501


        :return: The dimension of this Project.  # noqa: E501
        :rtype: str
        """
        return self._dimension

    @dimension.setter
    def dimension(self, dimension):
        """Sets the dimension of this Project.


        :param dimension: The dimension of this Project.  # noqa: E501
        :type: str
        """

        self._dimension = dimension

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(Project, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, Project):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
