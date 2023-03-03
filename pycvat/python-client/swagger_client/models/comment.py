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

class Comment(object):
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
        'id': 'int',
        'author': 'BasicUser',
        'author_id': 'int',
        'message': 'str',
        'created_date': 'datetime',
        'updated_date': 'datetime',
        'issue': 'int'
    }

    attribute_map = {
        'id': 'id',
        'author': 'author',
        'author_id': 'author_id',
        'message': 'message',
        'created_date': 'created_date',
        'updated_date': 'updated_date',
        'issue': 'issue'
    }

    def __init__(self, id=None, author=None, author_id=None, message=None, created_date=None, updated_date=None, issue=None):  # noqa: E501
        """Comment - a model defined in Swagger"""  # noqa: E501
        self._id = None
        self._author = None
        self._author_id = None
        self._message = None
        self._created_date = None
        self._updated_date = None
        self._issue = None
        self.discriminator = None
        if id is not None:
            self.id = id
        if author is not None:
            self.author = author
        if author_id is not None:
            self.author_id = author_id
        if message is not None:
            self.message = message
        if created_date is not None:
            self.created_date = created_date
        if updated_date is not None:
            self.updated_date = updated_date
        self.issue = issue

    @property
    def id(self):
        """Gets the id of this Comment.  # noqa: E501


        :return: The id of this Comment.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this Comment.


        :param id: The id of this Comment.  # noqa: E501
        :type: int
        """

        self._id = id

    @property
    def author(self):
        """Gets the author of this Comment.  # noqa: E501


        :return: The author of this Comment.  # noqa: E501
        :rtype: BasicUser
        """
        return self._author

    @author.setter
    def author(self, author):
        """Sets the author of this Comment.


        :param author: The author of this Comment.  # noqa: E501
        :type: BasicUser
        """

        self._author = author

    @property
    def author_id(self):
        """Gets the author_id of this Comment.  # noqa: E501


        :return: The author_id of this Comment.  # noqa: E501
        :rtype: int
        """
        return self._author_id

    @author_id.setter
    def author_id(self, author_id):
        """Sets the author_id of this Comment.


        :param author_id: The author_id of this Comment.  # noqa: E501
        :type: int
        """

        self._author_id = author_id

    @property
    def message(self):
        """Gets the message of this Comment.  # noqa: E501


        :return: The message of this Comment.  # noqa: E501
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message):
        """Sets the message of this Comment.


        :param message: The message of this Comment.  # noqa: E501
        :type: str
        """

        self._message = message

    @property
    def created_date(self):
        """Gets the created_date of this Comment.  # noqa: E501


        :return: The created_date of this Comment.  # noqa: E501
        :rtype: datetime
        """
        return self._created_date

    @created_date.setter
    def created_date(self, created_date):
        """Sets the created_date of this Comment.


        :param created_date: The created_date of this Comment.  # noqa: E501
        :type: datetime
        """

        self._created_date = created_date

    @property
    def updated_date(self):
        """Gets the updated_date of this Comment.  # noqa: E501


        :return: The updated_date of this Comment.  # noqa: E501
        :rtype: datetime
        """
        return self._updated_date

    @updated_date.setter
    def updated_date(self, updated_date):
        """Sets the updated_date of this Comment.


        :param updated_date: The updated_date of this Comment.  # noqa: E501
        :type: datetime
        """

        self._updated_date = updated_date

    @property
    def issue(self):
        """Gets the issue of this Comment.  # noqa: E501


        :return: The issue of this Comment.  # noqa: E501
        :rtype: int
        """
        return self._issue

    @issue.setter
    def issue(self, issue):
        """Sets the issue of this Comment.


        :param issue: The issue of this Comment.  # noqa: E501
        :type: int
        """
        if issue is None:
            raise ValueError("Invalid value for `issue`, must not be `None`")  # noqa: E501

        self._issue = issue

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
        if issubclass(Comment, dict):
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
        if not isinstance(other, Comment):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
