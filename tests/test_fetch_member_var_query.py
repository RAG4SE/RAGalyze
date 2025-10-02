#!/usr/bin/env python3

"""Tests for :class:`FetchMemberVarQuery`."""

from __future__ import annotations

from typing import Dict, List
from unittest import TestCase

from ragalyze.agent import FetchMemberVarQuery


class FetchMemberVarQueryTests(TestCase):
    def setUp(self) -> None:
        self.query = FetchMemberVarQuery(debug=False)

    @staticmethod
    def _find_member(members: List[Dict[str, object]], name: str) -> Dict[str, object] | None:
        return next((member for member in members if member.get("name") == name), None)

    def _require_member(self, members: List[Dict[str, object]], name: str) -> Dict[str, object]:
        member = self._find_member(members, name)
        self.assertIsNotNone(member, f"Expected member '{name}'")
        return member or {}

    def _assert_member_equals(self, member: Dict[str, object], expected_name: str, expected_type: str, expected_access: str = "private") -> None:
        """Helper to assert member properties."""
        self.assertEqual(member["name"], expected_name)
        self.assertEqual(member["type"], expected_type)
        self.assertEqual(member["access"], expected_access)


    def test_empty_class_definition(self) -> None:
        class_definition = "class EmptyClass {}"
        result = self.query(class_definition)
        self.assertEqual(result, [])

    def test_cpp_class_with_simple_members(self) -> None:
        class_definition = """
class MyClass {
private:
    int x;
    double y;
public:
    std::string name;
protected:
    bool flag;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        # Check first member
        self.assertEqual(result[0]["name"], "x")
        self.assertEqual(result[0]["type"], "int")
        self.assertEqual(result[0]["access"], "private")
        self.assertFalse(result[0]["is_static"])
        self.assertFalse(result[0]["is_const"])

    def test_cpp_class_with_static_members(self) -> None:
        class_definition = """
class Config {
private:
    static int instance_count;
    static const std::string VERSION;
public:
    static bool debug_mode;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3)

        # Check static member
        static_member = self._require_member(result, "instance_count")
        self.assertTrue(static_member["is_static"])

        # Check static const member
        const_static_member = self._require_member(result, "VERSION")
        self.assertTrue(const_static_member["is_static"])
        self.assertTrue(const_static_member.get("is_const"))

    def test_cpp_class_with_default_values(self) -> None:
        class_definition = """
class Settings {
private:
    int timeout = 30;
    std::string server_name = "localhost";
    bool enabled = true;
public:
    double version = 1.0;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        # Check member with default value
        member_with_default = self._require_member(result, "timeout")
        self.assertEqual(member_with_default["default_value"], "30")

    def test_cpp_class_with_complex_types(self) -> None:
        class_definition = """
class Container {
private:
    std::vector<int>* data_ptr;
    const std::map<std::string, std::vector<double>>& config_ref;
    std::unique_ptr<Logger> logger;
public:
    std::shared_ptr<Database> db_connection;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        # Check pointer type
        ptr_member = self._require_member(result, "data_ptr")
        self.assertEqual(ptr_member["type"], "std::vector<int>*")

        # Check const reference type
        const_ref_member = self._require_member(result, "config_ref")
        self.assertEqual(const_ref_member["type"], "const std::map<std::string, std::vector<double>>&")
        self.assertTrue(const_ref_member["is_const"])

    def test_cpp_class_with_attributes(self) -> None:
        class_definition = """
class ModernClass {
private:
    [[nodiscard]] int internal_id;
    [[deprecated("Use new_field instead")]] std::string old_field;
    [[maybe_unused]] double temp_value;
public:
    [[no_unique_address]] EmptyObject empty;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        # Check member with attributes
        member_with_attrs = self._require_member(result, "internal_id")
        self.assertEqual(member_with_attrs["attributes"], ["nodiscard"])

    def test_cpp_template_class(self) -> None:
        class_definition = """
template<typename T, typename U = int>
class Container {
private:
    T data;
    U capacity;
    static constexpr int DEFAULT_CAPACITY = 100;
public:
    std::vector<T> elements;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        # Check template members
        template_member = self._require_member(result, "data")
        self.assertEqual(template_member["type"], "T")

    def test_cpp_class_with_bit_fields(self) -> None:
        class_definition = """
class Flags {
private:
    unsigned int ready : 1;
    unsigned int error : 1;
    unsigned int status : 6;
public:
    unsigned int reserved : 24;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        bit_field_member = self._require_member(result, "ready")
        self.assertEqual(bit_field_member["type"], "unsigned int")

    def test_cpp_class_with_constexpr_members(self) -> None:
        class_definition = """
class Constants {
public:
    static constexpr double PI = 3.14159;
    static constexpr int MAX_SIZE = 1024;
private:
    constexpr int default_value = 42;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3, msg=f"result={result}")

        constexpr_member = self._require_member(result, "PI")
        self.assertTrue(constexpr_member["is_static"])
        self.assertTrue(constexpr_member["is_const"])

    def test_cpp_class_with_volatile_members(self) -> None:
        class_definition = """
class HardwareRegister {
private:
    volatile int status_register;
    volatile unsigned char control_byte;
public:
    volatile long* data_pointer;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3)

        volatile_member = self._require_member(result, "status_register")
        self.assertEqual(volatile_member["type"], "volatile int")

    def test_cpp_class_with_reference_members(self) -> None:
        class_definition = """
class ReferenceHolder {
private:
    int& counter_ref;
    const std::string& name_ref;
public:
    double& value_ref = default_value;
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3)

        ref_member = self._require_member(result, "counter_ref")
        self.assertEqual(ref_member["type"], "int&")

    def test_python_class_with_members(self) -> None:
        class_definition = """
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self._age = age
        self.__secret = "private"
        self._items: List[str] = []
        self.is_active = True
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 5)

        # Check Python member with type annotation
        typed_member = self._require_member(result, "_items")
        self.assertEqual(typed_member["type"], "List[str]")

    def test_python_class_with_class_variables(self) -> None:
        class_definition = """
class Database:
    MAX_CONNECTIONS = 10
    _connection_count = 0
    __instance = None

    def __init__(self):
        self.connection = None
        self._config = {}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 5)

        static_member = self._require_member(result, "MAX_CONNECTIONS")
        self.assertTrue(static_member["is_static"])

    def test_python_class_with_properties(self) -> None:
        class_definition = """
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def area(self):
        return self._width * self._height

    @property
    def perimeter(self):
        return 2 * (self._width + self._height)
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 2)

    def test_python_dataclass(self) -> None:
        class_definition = """
from dataclasses import dataclass
from typing import List

@dataclass
class Employee:
    name: str
    age: int
    department: str
    skills: List[str]
    is_active: bool = True
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 5)

        data_member = self._require_member(result, "name")
        self.assertEqual(data_member["type"], "str")

    def test_python_class_with_slots(self) -> None:
        class_definition = """
class Point:
    __slots__ = ['x', 'y', 'z']

    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3)

    def test_java_class_with_members(self) -> None:
        class_definition = """
public class User {
    private String username;
    private int age;
    protected String email;
    public boolean isActive;
    private static final long serialVersionUID = 1L;
    public static final String DEFAULT_ROLE = "user";
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 6)

        # Check Java static final member
        static_final_member = self._require_member(result, "serialVersionUID")
        self.assertTrue(static_final_member["is_static"])
        self.assertTrue(static_final_member["is_const"])

    def test_java_class_with_annotations(self) -> None:
        class_definition = """
public class Product {
    @Id
    private Long id;

    @Column(name = "product_name")
    private String name;

    @ManyToOne
    @JoinColumn(name = "category_id")
    private Category category;

    @Transient
    private double calculatedPrice;
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        annotated_member = self._require_member(result, "id")
        self.assertEqual(annotated_member["attributes"], ["Id"])

    def test_java_enum_constants(self) -> None:
        class_definition = """
public enum Status {
    PENDING("Pending"),
    ACTIVE("Active"),
    INACTIVE("Inactive");

    private final String displayName;

    Status(String displayName) {
        this.displayName = displayName;
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 1)

        enum_member = self._require_member(result, "displayName")
        self.assertEqual(enum_member["type"], "String")
        self.assertTrue(enum_member["is_const"])

    def test_java_record_class(self) -> None:
        class_definition = """
public record Point(int x, int y, String label) {
    public static final Point ORIGIN = new Point(0, 0, "Origin");

    public Point withLabel(String newLabel) {
        return new Point(x, y, newLabel);
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        record_member = self._require_member(result, "x")
        self.assertEqual(record_member["type"], "int")

    def test_java_interface_with_constants(self) -> None:
        class_definition = """
public interface DatabaseConfig {
    String DEFAULT_URL = "jdbc:mysql://localhost:3306/mydb";
    int MAX_CONNECTIONS = 10;
    long TIMEOUT_MS = 5000L;
    boolean ENABLE_POOLING = true;
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        interface_member = self._require_member(result, "DEFAULT_URL")
        self.assertTrue(interface_member["is_static"])
        self.assertTrue(interface_member["is_const"])
        self.assertEqual(interface_member["type"], "String")

    def test_java_abstract_class(self) -> None:
        class_definition = """
public abstract class Animal {
    protected String name;
    private int age;
    public static final String SPECIES = "Unknown";

    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public abstract void makeSound();

    public String getName() {
        return name;
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3)

        protected_member = self._require_member(result, "name")
        self.assertEqual(protected_member["access"], "protected")

    # Solidity test cases
    def test_solidity_contract_with_state_variables(self) -> None:
        class_definition = """
contract Token {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    address public owner;
    bool public paused;

    constructor(string memory _name, string memory _symbol, uint8 _decimals) {
        owner = msg.sender;
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 7)

        token_member = self._require_member(result, "name")
        self.assertEqual(token_member["type"], "string")
        self.assertEqual(token_member["access"], "public")

    def test_solidity_contract_with_constants(self) -> None:
        class_definition = """
contract Constants {
    uint256 public constant MAX_SUPPLY = 1000000 * 10**18;
    address public immutable OWNER;
    string public constant VERSION = "1.0.0";
    bytes32 public constant DOMAIN_SEPARATOR = keccak256("DOMAIN");

    constructor() {
        OWNER = msg.sender;
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 4)

        constant_member = self._require_member(result, "MAX_SUPPLY")
        self.assertTrue(constant_member["is_const"])
        self.assertTrue(constant_member["is_static"])

    def test_solidity_struct_definition(self) -> None:
        class_definition = """
contract UserManagement {
    struct User {
        uint256 id;
        address wallet;
        string name;
        bool isActive;
        uint256 createdAt;
        UserRole role;
    }

    enum UserRole {
        ADMIN,
        USER,
        GUEST
    }

    mapping(uint256 => User) public users;
    mapping(address => uint256) public addressToUserId;
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 2)

        mapping_member = self._require_member(result, "users")
        self.assertEqual(mapping_member["type"], "mapping(uint256 => User)")

    def test_solidity_inheritance_contract(self) -> None:
        class_definition = """
contract Ownable {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
}

contract Pausable is Ownable {
    bool public paused;

    function pause() public onlyOwner {
        paused = true;
    }

    function unpause() public onlyOwner {
        paused = false;
    }
}

contract Token is Ownable, Pausable {
    string public name;
    mapping(address => uint256) public balances;

    constructor(string memory _name) {
        name = _name;
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3)

        token_name = self._require_member(result, "name")
        self.assertEqual(token_name["type"], "string")

    # Additional C++ test cases
    def test_cpp_class_with_friend_functions(self) -> None:
        class_definition = """
class Matrix {
private:
    double** data;
    int rows, cols;

public:
    Matrix(int r, int c) : rows(r), cols(c) {
        data = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            data[i] = new double[cols]();
        }
    }

    ~Matrix() {
        for (int i = 0; i < rows; ++i) {
            delete[] data[i];
        }
        delete[] data;
    }

    friend Matrix operator+(const Matrix& a, const Matrix& b);
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

    double& operator()(int i, int j) { return data[i][j]; }
    const double& operator()(int i, int j) const { return data[i][j]; }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 2)

        rows_member = self._require_member(result, "rows")
        self.assertEqual(rows_member["type"], "int")
        self.assertEqual(rows_member["access"], "private")

    def test_cpp_class_with_virtual_functions(self) -> None:
        class_definition = """
class Shape {
protected:
    std::string name;
    double x, y;

public:
    Shape(const std::string& n, double px, double py)
        : name(n), x(px), y(py) {}

    virtual ~Shape() = default;

    virtual double area() const = 0;
    virtual double perimeter() const = 0;

    virtual void move(double dx, double dy) {
        x += dx;
        y += dy;
    }

    virtual void scale(double factor) = 0;

    std::string getName() const { return name; }
    double getX() const { return x; }
    double getY() const { return y; }
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(const std::string& n, double px, double py, double r)
        : Shape(n, px, py), radius(r) {}

    double area() const override {
        return 3.14159 * radius * radius;
    }

    double perimeter() const override {
        return 2 * 3.14159 * radius;
    }

    void scale(double factor) override {
        radius *= factor;
    }

    double getRadius() const { return radius; }
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3)

        shape_name = self._require_member(result, "name")
        self.assertEqual(shape_name["type"], "std::string")
        self.assertEqual(shape_name["access"], "protected")

    def test_cpp_class_with_smart_pointers_and_move_semantics(self) -> None:
        class_definition = """
class ResourceManager {
private:
    std::unique_ptr<int[]> data;
    std::shared_ptr<std::vector<double>> cache;
    std::weak_ptr<ResourceManager> parent;
    mutable std::mutex mutex;
    std::atomic<size_t> ref_count;
    std::string resource_path;

public:
    ResourceManager(const std::string& path, size_t size)
        : resource_path(path), ref_count(0) {
        data = std::make_unique<int[]>(size);
        cache = std::make_shared<std::vector<double>>();
    }

    ResourceManager(ResourceManager&& other) noexcept
        : data(std::move(other.data)),
          cache(std::move(other.cache)),
          parent(std::move(other.parent)),
          ref_count(other.ref_count.load()),
          resource_path(std::move(other.resource_path)) {}

    ResourceManager& operator=(ResourceManager&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            cache = std::move(other.cache);
            parent = std::move(other.parent);
            ref_count = other.ref_count.load();
            resource_path = std::move(other.resource_path);
        }
        return *this;
    }

    void setParent(const std::shared_ptr<ResourceManager>& p) {
        parent = p;
    }

    std::shared_ptr<std::vector<double>> getCache() const {
        std::lock_guard<std::mutex> lock(mutex);
        return cache;
    }

    size_t incrementRefCount() {
        return ++ref_count;
    }

    const std::string& getPath() const { return resource_path; }
};
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 6)

        unique_ptr_member = self._require_member(result, "data")
        self.assertEqual(unique_ptr_member["type"], "std::unique_ptr<int[]>")

    # Additional Python test cases
    def test_python_class_with_descriptors(self) -> None:
        class_definition = """
class ValidatedAttribute:
    def __init__(self, name, validator=None):
        self.name = name
        self.validator = validator

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(f'_{self.name}')

    def __set__(self, obj, value):
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}: {value}")
        obj.__dict__[f'_{self.name}'] = value

class Person:
    name = ValidatedAttribute('name', lambda x: isinstance(x, str) and len(x) > 0)
    age = ValidatedAttribute('age', lambda x: isinstance(x, int) and x >= 0)
    email = ValidatedAttribute('email', lambda x: '@' in str(x))

    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

    @property
    def is_adult(self):
        return self.age >= 18

    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 7)

        descriptor_member = self._require_member(result, "name")
        self.assertEqual(descriptor_member["type"], "ValidatedAttribute")

    def test_python_class_with_metaclass(self) -> None:
        class_definition = """
class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=MetaSingleton):
    connection_string: str
    connection_pool: List[object]
    max_connections: int
    _is_connected: bool
    logger: logging.Logger

    def __init__(self, connection_string: str, max_connections: int = 10):
        self.connection_string = connection_string
        self.max_connections = max_connections
        self.connection_pool = []
        self._is_connected = False
        self.logger = logging.getLogger(__name__)

    def connect(self):
        if not self._is_connected:
            self._is_connected = True
            self.logger.info("Connected to database")

    def disconnect(self):
        if self._is_connected:
            self._is_connected = False
            self.logger.info("Disconnected from database")

    @classmethod
    def get_instance(cls):
        return cls()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 5)

        singleton_member = self._require_member(result, "connection_string")
        self.assertEqual(singleton_member["type"], "str")

    def test_python_class_with_generics_and_typevars(self) -> None:
        class_definition = """
from typing import TypeVar, Generic, List, Optional, Callable
from dataclasses import dataclass

T = TypeVar('T')
R = TypeVar('R')

@dataclass
class Container(Generic[T]):
    items: List[T]
    capacity: int
    filter_func: Optional[Callable[[T], bool]] = None

    def __init__(self, capacity: int = 10):
        self.items = []
        self.capacity = capacity
        self.filter_func = None

    def add(self, item: T) -> bool:
        if len(self.items) < self.capacity:
            if self.filter_func is None or self.filter_func(item):
                self.items.append(item)
                return True
        return False

    def remove(self, item: T) -> bool:
        if item in self.items:
            self.items.remove(item)
            return True
        return False

    def filter(self, predicate: Callable[[T], bool]) -> 'Container[T]':
        result = Container[T](self.capacity)
        result.items = [item for item in self.items if predicate(item)]
        return result

    def map(self, transform: Callable[[T], R]) -> 'Container[R]':
        result = Container[R](self.capacity)
        result.items = [transform(item) for item in self.items]
        return result

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item: T) -> bool:
        return item in self.items

    def __iter__(self):
        return iter(self.items)

    def __str__(self) -> str:
        return f"Container({self.items})"
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 3)

        generic_member = self._require_member(result, "items")
        self.assertEqual(generic_member["type"], "List[T]")

    # Additional Java test cases
    def test_java_generic_class_with_bounds(self) -> None:
        class_definition = """
import java.util.*;

public class GenericRepository<T extends Comparable<T>, ID> {
    private Map<ID, T> entities;
    private List<T> sortedEntities;
    private Comparator<T> comparator;
    private int maxCapacity;
    private final Class<T> entityClass;

    public GenericRepository(Class<T> entityClass, int maxCapacity) {
        this.entityClass = entityClass;
        this.maxCapacity = maxCapacity;
        this.entities = new HashMap<>();
        this.sortedEntities = new ArrayList<>();
        this.comparator = Comparator.naturalOrder();
    }

    public void add(ID id, T entity) {
        if (entities.size() >= maxCapacity) {
            throw new IllegalStateException("Repository is full");
        }
        entities.put(id, entity);
        updateSortedEntities();
    }

    public Optional<T> findById(ID id) {
        return Optional.ofNullable(entities.get(id));
    }

    public List<T> findAllSorted() {
        return new ArrayList<>(sortedEntities);
    }

    public List<T> findInRange(T min, T max) {
        return sortedEntities.stream()
            .filter(e -> e.compareTo(min) >= 0 && e.compareTo(max) <= 0)
            .collect(Collectors.toList());
    }

    private void updateSortedEntities() {
        sortedEntities = new ArrayList<>(entities.values());
        Collections.sort(sortedEntities, comparator);
    }

    public int size() {
        return entities.size();
    }

    public boolean isEmpty() {
        return entities.isEmpty();
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 5)

        generic_member = self._require_member(result, "entities")
        self.assertEqual(generic_member["type"], "Map<ID, T>")

    def test_java_class_with_enums_and_collections(self) -> None:
        class_definition = """
import java.time.*;
import java.util.*;

public class User {
    public enum Status {
        ACTIVE,
        INACTIVE,
        SUSPENDED,
        PENDING_VERIFICATION
    }

    public enum Role {
        ADMIN,
        USER,
        MODERATOR,
        GUEST
    }

    private UUID id;
    private String username;
    private String email;
    private String hashedPassword;
    private Set<Role> roles;
    private Status status;
    private LocalDateTime createdAt;
    private LocalDateTime lastLoginAt;
    private Map<String, String> metadata;
    private List<String> loginHistory;

    public User(String username, String email, String hashedPassword) {
        this.id = UUID.randomUUID();
        this.username = username;
        this.email = email;
        this.hashedPassword = hashedPassword;
        this.roles = new HashSet<>(Arrays.asList(Role.USER));
        this.status = Status.PENDING_VERIFICATION;
        this.createdAt = LocalDateTime.now();
        this.metadata = new HashMap<>();
        this.loginHistory = new ArrayList<>();
    }

    public void addRole(Role role) {
        roles.add(role);
    }

    public void removeRole(Role role) {
        roles.remove(role);
    }

    public boolean hasRole(Role role) {
        return roles.contains(role);
    }

    public void updateLastLogin() {
        this.lastLoginAt = LocalDateTime.now();
        this.loginHistory.add(lastLoginAt.toString());
    }

    public void addMetadata(String key, String value) {
        metadata.put(key, value);
    }

    public Optional<String> getMetadata(String key) {
        return Optional.ofNullable(metadata.get(key));
    }

    public boolean isActive() {
        return status == Status.ACTIVE;
    }

    public void activate() {
        this.status = Status.ACTIVE;
    }

    public void suspend() {
        this.status = Status.SUSPENDED;
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 10)

        enum_member = self._require_member(result, "roles")
        self.assertEqual(enum_member["type"], "Set<Role>")

    # Additional Solidity test cases
    def test_solidity_contract_with_events_and_modifiers(self) -> None:
        class_definition = """
contract AccessControl {
    event RoleGranted(address indexed account, string role, address indexed grantor);
    event RoleRevoked(address indexed account, string role, address indexed revoker);

    mapping(address => mapping(string => bool)) public roles;
    mapping(string => address) public roleAdmins;

    modifier onlyRole(string memory role) {
        require(roles[msg.sender][role], "AccessControl: account does not have role");
        _;
    }

    modifier onlyRoleAdmin(string memory role) {
        require(roleAdmins[role] == msg.sender, "AccessControl: not role admin");
        _;
    }

    constructor() {
        _setupRole("DEFAULT_ADMIN", msg.sender);
    }

    function grantRole(address account, string memory role) public onlyRoleAdmin(role) {
        _grantRole(account, role);
    }

    function revokeRole(address account, string memory role) public onlyRoleAdmin(role) {
        _revokeRole(account, role);
    }

    function renounceRole(address account, string memory role) public {
        require(account == msg.sender, "AccessControl: can only renounce roles for self");
        _revokeRole(account, role);
    }

    function _grantRole(address account, string memory role) internal {
        if (!roles[account][role]) {
            roles[account][role] = true;
            emit RoleGranted(account, role, msg.sender);
        }
    }

    function _revokeRole(address account, string memory role) internal {
        if (roles[account][role]) {
            roles[account][role] = false;
            emit RoleRevoked(account, role, msg.sender);
        }
    }

    function _setupRole(string memory role, address admin) internal {
        roleAdmins[role] = admin;
        _grantRole(admin, role);
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 2)

        role_mapping = self._require_member(result, "roles")
        self.assertEqual(role_mapping["type"], "mapping(address => mapping(string => bool))")

    def test_solidity_contract_with_libraries_and_safe_math(self) -> None:
        class_definition = """
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
        uint256 c = a / b;
        return c;
    }
}

contract Token {
    using SafeMath for uint256;

    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory _name, string memory _symbol, uint8 _decimals) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
    }

    function transfer(address to, uint256 value) public returns (bool) {
        require(to != address(0), "Token: transfer to the zero address");
        balanceOf[msg.sender] = balanceOf[msg.sender].sub(value);
        balanceOf[to] = balanceOf[to].add(value);
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public returns (bool) {
        require(spender != address(0), "Token: approve to the zero address");
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        require(to != address(0), "Token: transfer to the zero address");
        require(from != address(0), "Token: transfer from the zero address");
        allowance[from][msg.sender] = allowance[from][msg.sender].sub(value);
        balanceOf[from] = balanceOf[from].sub(value);
        balanceOf[to] = balanceOf[to].add(value);
        emit Transfer(from, to, value);
        return true;
    }
}
"""
        result = self.query(class_definition)
        self.assertEqual(len(result), 6)

        token_name = self._require_member(result, "name")
        self.assertEqual(token_name["type"], "string")

    # Test cases for target_name functionality
    def test_target_name_existing_member(self) -> None:
        """Test fetching a specific member variable that exists."""
        class_definition = """
class TestClass {
private:
    int private_var;
    std::string string_var;
public:
    static int static_var;
    const double const_var = 3.14;
};
"""
        result = self.query(class_definition, target_name="private_var")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "private_var")
        self.assertEqual(result["type"], "int")
        self.assertEqual(result["access"], "private")

    def test_target_name_nonexistent_member(self) -> None:
        """Test fetching a specific member variable that doesn't exist."""
        class_definition = """
class TestClass {
private:
    int existing_var;
public:
    std::string public_var;
};
"""
        result = self.query(class_definition, target_name="nonexistent_var")
        self.assertIsNone(result)

    def test_target_name_with_empty_class(self) -> None:
        """Test target_name functionality with empty class definition."""
        class_definition = "class EmptyClass {}"
        result = self.query(class_definition, target_name="any_var")
        self.assertIsNone(result)

    def test_target_name_case_sensitivity(self) -> None:
        """Test that target_name matching is case sensitive."""
        class_definition = """
class TestCase {
private:
    int Value;
    int value;
    int VALUE;
};
"""
        # Test exact case match
        result_exact = self.query(class_definition, target_name="Value")
        self.assertIsNotNone(result_exact)
        self.assertEqual(result_exact["name"], "Value")

        # Test different case
        result_diff = self.query(class_definition, target_name="value")
        self.assertIsNotNone(result_diff)
        self.assertEqual(result_diff["name"], "value")

        # Test another case
        result_upper = self.query(class_definition, target_name="VALUE")
        self.assertIsNotNone(result_upper)
        self.assertEqual(result_upper["name"], "VALUE")

    def test_target_name_with_special_characters(self) -> None:
        """Test target_name functionality with special characters in member names."""
        class_definition = """
class SpecialNames {
private:
    int member_with_underscores;
    int member123;
    std::string member_with_numbers_123;
};
"""
        result = self.query(class_definition, target_name="member_with_underscores")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "member_with_underscores")

        result_nums = self.query(class_definition, target_name="member123")
        self.assertIsNotNone(result_nums)
        self.assertEqual(result_nums["name"], "member123")

    def test_target_name_vs_all_members_consistency(self) -> None:
        """Test that target_name result is consistent with result from all members."""
        class_definition = """
class ConsistencyTest {
private:
    int target_member;
    std::string other_member;
public:
    double public_member;
};
"""
        # Get all members
        all_members = self.query(class_definition)
        target_from_all = next((m for m in all_members if m["name"] == "target_member"), None)

        # Get specific member
        target_direct = self.query(class_definition, target_name="target_member")

        # Compare results
        self.assertIsNotNone(target_from_all)
        self.assertIsNotNone(target_direct)
        self.assertEqual(target_from_all, target_direct)

    def test_target_name_python_class(self) -> None:
        """Test target_name functionality with Python class."""
        class_definition = """
class PythonClass:
    def __init__(self):
        self.private_var = "private"
        self._protected_var = "protected"
        self.public_var = "public"
        self.__very_private = "very_private"
"""
        result = self.query(class_definition, target_name="private_var")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "private_var")

    def test_target_name_java_class(self) -> None:
        """Test target_name functionality with Java class."""
        class_definition = """
public class JavaClass {
    private String privateField;
    protected int protectedField;
    public double publicField;
    static final String CONSTANT = "TEST";
}
"""
        result = self.query(class_definition, target_name="CONSTANT")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "CONSTANT")
        self.assertTrue(result["is_static"])
        self.assertTrue(result["is_const"])

    def test_target_name_solidity_contract(self) -> None:
        """Test target_name functionality with Solidity contract."""
        class_definition = """
contract SolidityContract {
    string public name;
    uint256 public value;
    address public owner;
    bool private internal_flag;
}
"""
        result = self.query(class_definition, target_name="owner")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "owner")
        self.assertEqual(result["type"], "address")

    def test_target_name_with_duplicates(self) -> None:
        """Test target_name when multiple members have similar names."""
        class_definition = """
class DuplicateNames {
private:
    int count;
    int count_backup;
    int count_temp;
public:
    double max_count;
};
"""
        # Should return exact match
        result = self.query(class_definition, target_name="count")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "count")
        self.assertEqual(result["type"], "int")

        # Should return different exact match
        result_backup = self.query(class_definition, target_name="count_backup")
        self.assertIsNotNone(result_backup)
        self.assertEqual(result_backup["name"], "count_backup")


if __name__ == "__main__":
    import unittest
    unittest.main()
