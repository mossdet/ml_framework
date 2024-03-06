

class Vehicle():
    def __init__(self, make:str, model:str, color:str):
        """
        initializes a new instance of the Vehicle class

        Args:
            make (str): the make of the vehicle
            model (str): the model of the vehicle
            color (str): the color of the vehicle
        """
        self.make = make
        self.model = model
        self.color = color
        self.position = 0
    
    def move_forward(self, steps: int):
        """
        move the vehicle forward a certain number of steps

        Args:
            steps (int): the number of steps to move the vehicle
        """
        self.position = self.position + steps

    def print_position(self):
        """
        print the current position of the vehicle
        """
        print(self.position)
    
    # Class methods are common to all objects of this class. the cls input parameter is similar to the self parameter
    # cls is a pointer to the address in memory of the class
    @classmethod
    def get_number_of_wheels(cls):
        """
        get the number of wheels for the class

        Returns:
            int: the number of wheels for the class
        """
        return 1


class Car(Vehicle):
    def __init__(self, make: str, model: str, color: str):
        super().__init__(make=make, model=model, color=color)

    @classmethod
    def get_number_of_wheels(cls):
        """
        get the number of wheels for the class

        Returns:
            int: the number of wheels for the class
        """
        return 4


class Motorcycle(Vehicle):
    def __init__(self, make: str, model: str, color: str, cc: int):
        super().__init__(make=make, model=model, color=color)
        self.cc = cc

    def get_cc(self):
        """
        get the cc of the motorcycle

        Returns:
            int: the cc of the motorcycle
        """
        return self.cc

        @classmethod
    def get_number_of_wheels(cls):
        """
        get the number of wheels for the class

        Returns:
            int: the number of wheels for the class
        """
        return 2



if __name__ == "__main__":
    a = Vehicle(make="mazda", model="six", color="blue")
    b = Vehicle(make="honda", model="pilot", color="red")

    a.print_position()
    a.move_forward(steps=2)
    a.print_position()

    b.print_position()
    b.move_forward(steps=5)
    b.print_position()
    print(Vehicle.get_number_of_wheels())

    toyota = Car(make="toyota", model="camry", color="green")
    toyota.move_forward(steps=7)
    toyota.print_position()
    print(Car.get_number_of_wheels())

    if toyota.get_number_of_wheels()==4:
        print("This is a car")

    suzuki = Motorcycle(make="suzuki", model="zx", color="black", cc=1000)
    print(suzuki.get_cc())
