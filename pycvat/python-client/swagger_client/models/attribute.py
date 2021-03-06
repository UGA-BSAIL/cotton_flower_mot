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


class Attribute(object):
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
        "mutable": "bool",
        "input_type": "str",
        "default_value": "str",
        "values": "str",
    }

    attribute_map = {
        "id": "id",
        "name": "name",
        "mutable": "mutable",
        "input_type": "input_type",
        "default_value": "default_value",
        "values": "values",
    }

    def __init__(
        self,
        id=None,
        name=None,
        mutable=None,
        input_type=None,
        default_value=None,
        values=None,
    ):  # noqa: E501
        """Attribute - a model defined in Swagger"""  # noqa: E501
        self._id = None
        self._name = None
        self._mutable = None
        self._input_type = None
        self._default_value = None
        self._values = None
        self.discriminator = None
        if id is not None:
            self.id = id
        self.name = name
        self.mutable = mutable
        self.input_type = input_type
        self.default_value = default_value
        self.values = values

    @property
    def id(self):
        """Gets the id of this Attribute.  # noqa: E501


        :return: The id of this Attribute.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this Attribute.


        :param id: The id of this Attribute.  # noqa: E501
        :type: int
        """

        self._id = id

    @property
    def name(self):
        """Gets the name of this Attribute.  # noqa: E501


        :return: The name of this Attribute.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this Attribute.


        :param name: The name of this Attribute.  # noqa: E501
        :type: str
        """
        if name is None:
            raise ValueError(
                "Invalid value for `name`, must not be `None`"
            )  # noqa: E501

        self._name = name

    @property
    def mutable(self):
        """Gets the mutable of this Attribute.  # noqa: E501


        :return: The mutable of this Attribute.  # noqa: E501
        :rtype: bool
        """
        return self._mutable

    @mutable.setter
    def mutable(self, mutable):
        """Sets the mutable of this Attribute.


        :param mutable: The mutable of this Attribute.  # noqa: E501
        :type: bool
        """
        if mutable is None:
            raise ValueError(
                "Invalid value for `mutable`, must not be `None`"
            )  # noqa: E501

        self._mutable = mutable

    @property
    def input_type(self):
        """Gets the input_type of this Attribute.  # noqa: E501


        :return: The input_type of this Attribute.  # noqa: E501
        :rtype: str
        """
        return self._input_type

    @input_type.setter
    def input_type(self, input_type):
        """Sets the input_type of this Attribute.


        :param input_type: The input_type of this Attribute.  # noqa: E501
        :type: str
        """
        if input_type is None:
            raise ValueError(
                "Invalid value for `input_type`, must not be `None`"
            )  # noqa: E501
        allowed_values = [
            "checkbox",
            "radio",
            "number",
            "text",
            "select",
        ]  # noqa: E501
        if input_type not in allowed_values:
            raise ValueError(
                "Invalid value for `input_type` ({0}), must be one of {1}".format(  # noqa: E501
                    input_type, allowed_values
                )
            )

        self._input_type = input_type

    @property
    def default_value(self):
        """Gets the default_value of this Attribute.  # noqa: E501


        :return: The default_value of this Attribute.  # noqa: E501
        :rtype: str
        """
        return self._default_value

    @default_value.setter
    def default_value(self, default_value):
        """Sets the default_value of this Attribute.


        :param default_value: The default_value of this Attribute.  # noqa: E501
        :type: str
        """
        if default_value is None:
            raise ValueError(
                "Invalid value for `default_value`, must not be `None`"
            )  # noqa: E501

        self._default_value = default_value

    @property
    def values(self):
        """Gets the values of this Attribute.  # noqa: E501


        :return: The values of this Attribute.  # noqa: E501
        :rtype: str
        """
        return self._values

    @values.setter
    def values(self, values):
        """Sets the values of this Attribute.


        :param values: The values of this Attribute.  # noqa: E501
        :type: str
        """
        if values is None:
            raise ValueError(
                "Invalid value for `values`, must not be `None`"
            )  # noqa: E501

        self._values = values

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
        if issubclass(Attribute, dict):
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
        if not isinstance(other, Attribute):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
