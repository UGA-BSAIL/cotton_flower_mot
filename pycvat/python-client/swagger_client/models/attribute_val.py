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


class AttributeVal(object):
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
    swagger_types = {"spec_id": "int", "value": "str"}

    attribute_map = {"spec_id": "spec_id", "value": "value"}

    def __init__(self, spec_id=None, value=None):  # noqa: E501
        """AttributeVal - a model defined in Swagger"""  # noqa: E501
        self._spec_id = None
        self._value = None
        self.discriminator = None
        self.spec_id = spec_id
        self.value = value

    @property
    def spec_id(self):
        """Gets the spec_id of this AttributeVal.  # noqa: E501


        :return: The spec_id of this AttributeVal.  # noqa: E501
        :rtype: int
        """
        return self._spec_id

    @spec_id.setter
    def spec_id(self, spec_id):
        """Sets the spec_id of this AttributeVal.


        :param spec_id: The spec_id of this AttributeVal.  # noqa: E501
        :type: int
        """
        if spec_id is None:
            raise ValueError(
                "Invalid value for `spec_id`, must not be `None`"
            )  # noqa: E501

        self._spec_id = spec_id

    @property
    def value(self):
        """Gets the value of this AttributeVal.  # noqa: E501


        :return: The value of this AttributeVal.  # noqa: E501
        :rtype: str
        """
        return self._value

    @value.setter
    def value(self, value):
        """Sets the value of this AttributeVal.


        :param value: The value of this AttributeVal.  # noqa: E501
        :type: str
        """
        if value is None:
            raise ValueError(
                "Invalid value for `value`, must not be `None`"
            )  # noqa: E501

        self._value = value

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
        if issubclass(AttributeVal, dict):
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
        if not isinstance(other, AttributeVal):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
