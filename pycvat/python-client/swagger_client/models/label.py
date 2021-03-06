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


class Label(object):
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
        "id": "int",
        "name": "str",
        "color": "str",
        "attributes": "list[Attribute]",
        "deleted": "bool",
    }

    attribute_map = {
        "id": "id",
        "name": "name",
        "color": "color",
        "attributes": "attributes",
        "deleted": "deleted",
    }

    def __init__(
        self, id=None, name=None, color=None, attributes=None, deleted=None
    ):  # noqa: E501
        """Label - a model defined in Swagger"""  # noqa: E501
        self._id = None
        self._name = None
        self._color = None
        self._attributes = None
        self._deleted = None
        self.discriminator = None
        if id is not None:
            self.id = id
        self.name = name
        if color is not None:
            self.color = color
        if attributes is not None:
            self.attributes = attributes
        if deleted is not None:
            self.deleted = deleted

    @property
    def id(self):
        """Gets the id of this Label.  # noqa: E501


        :return: The id of this Label.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this Label.


        :param id: The id of this Label.  # noqa: E501
        :type: int
        """

        self._id = id

    @property
    def name(self):
        """Gets the name of this Label.  # noqa: E501


        :return: The name of this Label.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this Label.


        :param name: The name of this Label.  # noqa: E501
        :type: str
        """
        if name is None:
            raise ValueError(
                "Invalid value for `name`, must not be `None`"
            )  # noqa: E501

        self._name = name

    @property
    def color(self):
        """Gets the color of this Label.  # noqa: E501


        :return: The color of this Label.  # noqa: E501
        :rtype: str
        """
        return self._color

    @color.setter
    def color(self, color):
        """Sets the color of this Label.


        :param color: The color of this Label.  # noqa: E501
        :type: str
        """

        self._color = color

    @property
    def attributes(self):
        """Gets the attributes of this Label.  # noqa: E501


        :return: The attributes of this Label.  # noqa: E501
        :rtype: list[Attribute]
        """
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        """Sets the attributes of this Label.


        :param attributes: The attributes of this Label.  # noqa: E501
        :type: list[Attribute]
        """

        self._attributes = attributes

    @property
    def deleted(self):
        """Gets the deleted of this Label.  # noqa: E501

        Delete label if value is true from proper Task/Project object  # noqa: E501

        :return: The deleted of this Label.  # noqa: E501
        :rtype: bool
        """
        return self._deleted

    @deleted.setter
    def deleted(self, deleted):
        """Sets the deleted of this Label.

        Delete label if value is true from proper Task/Project object  # noqa: E501

        :param deleted: The deleted of this Label.  # noqa: E501
        :type: bool
        """

        self._deleted = deleted

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(
                    map(
                        lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                        value,
                    )
                )
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value
        if issubclass(Label, dict):
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
        if not isinstance(other, Label):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
