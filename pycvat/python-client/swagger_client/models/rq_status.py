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


class RqStatus(object):
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
    swagger_types = {"state": "str", "message": "str"}

    attribute_map = {"state": "state", "message": "message"}

    def __init__(self, state=None, message=""):  # noqa: E501
        """RqStatus - a model defined in Swagger"""  # noqa: E501
        self._state = None
        self._message = None
        self.discriminator = None
        self.state = state
        if message is not None:
            self.message = message

    @property
    def state(self):
        """Gets the state of this RqStatus.  # noqa: E501


        :return: The state of this RqStatus.  # noqa: E501
        :rtype: str
        """
        return self._state

    @state.setter
    def state(self, state):
        """Sets the state of this RqStatus.


        :param state: The state of this RqStatus.  # noqa: E501
        :type: str
        """
        if state is None:
            raise ValueError(
                "Invalid value for `state`, must not be `None`"
            )  # noqa: E501
        allowed_values = [
            "Queued",
            "Started",
            "Finished",
            "Failed",
        ]  # noqa: E501
        if state not in allowed_values:
            raise ValueError(
                "Invalid value for `state` ({0}), must be one of {1}".format(  # noqa: E501
                    state, allowed_values
                )
            )

        self._state = state

    @property
    def message(self):
        """Gets the message of this RqStatus.  # noqa: E501


        :return: The message of this RqStatus.  # noqa: E501
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message):
        """Sets the message of this RqStatus.


        :param message: The message of this RqStatus.  # noqa: E501
        :type: str
        """

        self._message = message

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
        if issubclass(RqStatus, dict):
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
        if not isinstance(other, RqStatus):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
