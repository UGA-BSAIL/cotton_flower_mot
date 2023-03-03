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

class TrainingProject(object):
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
        'host': 'str',
        'username': 'str',
        'password': 'str',
        'enabled': 'bool',
        'project_class': 'str'
    }

    attribute_map = {
        'host': 'host',
        'username': 'username',
        'password': 'password',
        'enabled': 'enabled',
        'project_class': 'project_class'
    }

    def __init__(self, host=None, username=None, password=None, enabled=None, project_class=None):  # noqa: E501
        """TrainingProject - a model defined in Swagger"""  # noqa: E501
        self._host = None
        self._username = None
        self._password = None
        self._enabled = None
        self._project_class = None
        self.discriminator = None
        self.host = host
        self.username = username
        self.password = password
        if enabled is not None:
            self.enabled = enabled
        if project_class is not None:
            self.project_class = project_class

    @property
    def host(self):
        """Gets the host of this TrainingProject.  # noqa: E501


        :return: The host of this TrainingProject.  # noqa: E501
        :rtype: str
        """
        return self._host

    @host.setter
    def host(self, host):
        """Sets the host of this TrainingProject.


        :param host: The host of this TrainingProject.  # noqa: E501
        :type: str
        """
        if host is None:
            raise ValueError("Invalid value for `host`, must not be `None`")  # noqa: E501

        self._host = host

    @property
    def username(self):
        """Gets the username of this TrainingProject.  # noqa: E501


        :return: The username of this TrainingProject.  # noqa: E501
        :rtype: str
        """
        return self._username

    @username.setter
    def username(self, username):
        """Sets the username of this TrainingProject.


        :param username: The username of this TrainingProject.  # noqa: E501
        :type: str
        """
        if username is None:
            raise ValueError("Invalid value for `username`, must not be `None`")  # noqa: E501

        self._username = username

    @property
    def password(self):
        """Gets the password of this TrainingProject.  # noqa: E501


        :return: The password of this TrainingProject.  # noqa: E501
        :rtype: str
        """
        return self._password

    @password.setter
    def password(self, password):
        """Sets the password of this TrainingProject.


        :param password: The password of this TrainingProject.  # noqa: E501
        :type: str
        """
        if password is None:
            raise ValueError("Invalid value for `password`, must not be `None`")  # noqa: E501

        self._password = password

    @property
    def enabled(self):
        """Gets the enabled of this TrainingProject.  # noqa: E501


        :return: The enabled of this TrainingProject.  # noqa: E501
        :rtype: bool
        """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        """Sets the enabled of this TrainingProject.


        :param enabled: The enabled of this TrainingProject.  # noqa: E501
        :type: bool
        """

        self._enabled = enabled

    @property
    def project_class(self):
        """Gets the project_class of this TrainingProject.  # noqa: E501


        :return: The project_class of this TrainingProject.  # noqa: E501
        :rtype: str
        """
        return self._project_class

    @project_class.setter
    def project_class(self, project_class):
        """Sets the project_class of this TrainingProject.


        :param project_class: The project_class of this TrainingProject.  # noqa: E501
        :type: str
        """
        allowed_values = ["OD"]  # noqa: E501
        if project_class not in allowed_values:
            raise ValueError(
                "Invalid value for `project_class` ({0}), must be one of {1}"  # noqa: E501
                .format(project_class, allowed_values)
            )

        self._project_class = project_class

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
        if issubclass(TrainingProject, dict):
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
        if not isinstance(other, TrainingProject):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
