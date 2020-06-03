import msvcrt


def manual_controller():
    action = None
    while action is None:
        key = msvcrt.getwch()
        if key == 'w':
            action = 1
        elif key == 'a':
            action = 2
        elif key == 'd':
            action = 3
        elif key == 'z':
            action = 5
        elif key == 'x':
            action = 4
        elif key == 's':
            action = 0
        else:
            print("Invalid key")
    return action
