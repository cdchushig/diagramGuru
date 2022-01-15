from enum import Enum


class errorcode(Enum):
    success = 0
    warning = 1
    invalid = 2

# Print Enum member "success" for class "errorcode" using string format
print('Exit message: {}'.format(errorcode.success))

# Print Enum member "invalid" for class "errorcode"
print('Exit message:', errorcode.invalid)

# Print Enum member "warning" for class "errorcode"
print('Exit message:', errorcode(1))