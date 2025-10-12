// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Complex Vulnerable Token
 * @dev A complex token contract with multiple vulnerability patterns
 * Contains: Reentrancy, Integer Overflow, Access Control, Logic Flaws
 */

interface IComplexToken {
    function transfer(address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function mint(address to, uint256 amount) external;
    function burn(uint256 amount) external;
}

contract ComplexToken is IComplexToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => bool) public frozenAccounts;
    mapping(address => uint256) public lastTransferTime;
    mapping(address => uint256) public transferLimits;

    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    uint256 public maxSupply;

    address public owner;
    address public minter;
    address public freezer;

    uint256 public transferFee = 10; // 1% in basis points
    uint256 public burnRate = 5; // 0.5% in basis points

    event Transfer(address indexed from, address indexed to, uint256 amount);
    event Approval(address indexed owner, address indexed spender, uint256 amount);
    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);
    event AccountFrozen(address indexed account);
    event AccountUnfrozen(address indexed account);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyMinter() {
        require(msg.sender == minter, "Not minter");
        _;
    }

    modifier onlyFreezer() {
        require(msg.sender == freezer, "Not freezer");
        _;
    }

    modifier notFrozen(address account) {
        require(!frozenAccounts[account], "Account frozen");
        _;
    }

    constructor(string memory _name, string memory _symbol, uint8 _decimals, uint256 _maxSupply) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
        maxSupply = _maxSupply;
        owner = msg.sender;
        minter = msg.sender;
        freezer = msg.sender;
    }

    /**
     * @dev VULNERABILITY: Reentrancy in transfer function
     */
    function transfer(address to, uint256 amount) external override notFrozen(msg.sender) notFrozen(to) returns (bool) {
        require(amount > 0, "Amount must be > 0");
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");

        // Calculate fee
        uint256 fee = (amount * transferFee) / 1000;
        uint256 burnAmount = (amount * burnRate) / 1000;
        uint256 transferAmount = amount - fee - burnAmount;

        // Vulnerable: state change after external call
        (bool success, ) = msg.sender.call{value: 0}(""); // Could trigger reentrancy
        require(success, "Call failed");

        // Update balances
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += transferAmount;
        balanceOf[address(0)] += burnAmount;

        lastTransferTime[msg.sender] = block.timestamp;

        emit Transfer(msg.sender, to, transferAmount);
        emit Transfer(msg.sender, address(0), burnAmount);

        return true;
    }

    /**
     * @dev VULNERABILITY: Integer overflow in approve function
     */
    function approve(address spender, uint256 amount) external override notFrozen(msg.sender) notFrozen(spender) returns (bool) {
        // Vulnerable: potential overflow
        allowance[msg.sender][spender] += amount;

        emit Approval(msg.sender, spender, allowance[msg.sender][spender]);
        return true;
    }

    /**
     * @dev VULNERABILITY: Unchecked external call in transferFrom
     */
    function transferFrom(address from, address to, uint256 amount) external override notFrozen(from) notFrozen(to) notFrozen(msg.sender) returns (bool) {
        require(amount > 0, "Amount must be > 0");
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Insufficient allowance");

        // Check transfer limits
        if (transferLimits[from] > 0) {
            require(amount <= transferLimits[from], "Exceeds transfer limit");
        }

        // Check time-based transfer limits
        if (lastTransferTime[from] > 0 && block.timestamp - lastTransferTime[from] < 1 hours) {
            require(amount <= balanceOf[from] / 10, "Exceeds hourly limit");
        }

        // Calculate fee
        uint256 fee = (amount * transferFee) / 1000;
        uint256 burnAmount = (amount * burnRate) / 1000;
        uint256 transferAmount = amount - fee - burnAmount;

        // Vulnerable: unchecked external call
        (bool success, ) = from.call{value: 0}(""); // Could be used for reentrancy
        require(success, "Call failed");

        // Update balances
        balanceOf[from] -= amount;
        balanceOf[to] += transferAmount;
        balanceOf[address(0)] += burnAmount;
        allowance[from][msg.sender] -= amount;

        lastTransferTime[from] = block.timestamp;

        emit Transfer(from, to, transferAmount);
        emit Transfer(from, address(0), burnAmount);

        return true;
    }

    /**
     * @dev VULNERABILITY: Mint without supply validation
     */
    function mint(address to, uint256 amount) external override onlyMinter {
        // Vulnerable: no max supply check
        balanceOf[to] += amount;
        totalSupply += amount;

        emit Mint(to, amount);
        emit Transfer(address(0), to, amount);
    }

    /**
     * @dev VULNERABILITY: Integer underflow in burn
     */
    function burn(uint256 amount) external override notFrozen(msg.sender) {
        require(amount > 0, "Amount must be > 0");
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");

        // Vulnerable: potential underflow if totalSupply < amount
        totalSupply -= amount;
        balanceOf[msg.sender] -= amount;

        emit Burn(msg.sender, amount);
        emit Transfer(msg.sender, address(0), amount);
    }

    /**
     * @dev VULNERABILITY: Flash loan attack vulnerability
     */
    function flashLoan(uint256 amount, address borrower) external {
        require(amount > 0, "Amount must be > 0");
        require(balanceOf[address(this)] >= amount, "Insufficient liquidity");

        // Transfer tokens
        balanceOf[address(this)] -= amount;
        balanceOf[borrower] += amount;

        // Execute callback - vulnerable to reentrancy
        (bool success, ) = borrower.call(abi.encodeWithSignature("flashLoanCallback(uint256)", amount));
        require(success, "Flash loan callback failed");

        // Check repayment
        require(balanceOf[address(this)] >= amount, "Flash loan not repaid");
    }

    /**
     * @dev VULNERABILITY: Access control issue in critical function
     */
    function emergencyMint(address to, uint256 amount) external {
        // Vulnerable: no access control check
        balanceOf[to] += amount;
        totalSupply += amount;

        emit Mint(to, amount);
        emit Transfer(address(0), to, amount);
    }

    /**
     * @dev VULNERABILITY: Unchecked low-level call
     */
    function multiTransfer(address[] memory recipients, uint256[] memory amounts) external notFrozen(msg.sender) {
        require(recipients.length == amounts.length, "Arrays length mismatch");

        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }

        require(balanceOf[msg.sender] >= totalAmount, "Insufficient balance");

        for (uint256 i = 0; i < recipients.length; i++) {
            address recipient = recipients[i];
            uint256 amount = amounts[i];

            // Vulnerable: unchecked external call
            (bool success, ) = recipient.call{value: 0}(""); // Could trigger reentrancy
            require(success, "Call failed");

            balanceOf[msg.sender] -= amount;
            balanceOf[recipient] += amount;

            emit Transfer(msg.sender, recipient, amount);
        }
    }

    /**
     * @dev VULNERABILITY: Self-destruct vulnerability
     */
    function emergencyWithdraw() external onlyOwner {
        // Vulnerable: can be used to withdraw all funds
        selfdestruct(payable(owner));
    }

    /**
     * @dev VULNERABILITY: DoS with unbounded loop
     */
    function airdrop(address[] memory recipients, uint256 amountPerRecipient) external notFrozen(msg.sender) {
        require(amountPerRecipient > 0, "Amount must be > 0");
        uint256 totalAmount = amountPerRecipient * recipients.length;
        require(balanceOf[msg.sender] >= totalAmount, "Insufficient balance");

        // Vulnerable: unbounded loop
        for (uint256 i = 0; i < recipients.length; i++) {
            balanceOf[msg.sender] -= amountPerRecipient;
            balanceOf[recipients[i]] += amountPerRecipient;
            emit Transfer(msg.sender, recipients[i], amountPerRecipient);
        }
    }

    /**
     * @dev VULNERABILITY: Front-running vulnerability
     */
    function updateTransferFee(uint256 newFee) external onlyOwner {
        // Vulnerable: front-running possible
        transferFee = newFee;
    }

    function freezeAccount(address account) external onlyFreezer {
        frozenAccounts[account] = true;
        emit AccountFrozen(account);
    }

    function unfreezeAccount(address account) external onlyFreezer {
        frozenAccounts[account] = false;
        emit AccountUnfrozen(account);
    }

    function setTransferLimit(address account, uint256 limit) external onlyOwner {
        transferLimits[account] = limit;
    }

    function setMinter(address newMinter) external onlyOwner {
        minter = newMinter;
    }

    function setFreezer(address newFreezer) external onlyOwner {
        freezer = newFreezer;
    }

    /**
     * @dev VULNERABILITY: Arbitrary call through delegatecall
     */
    function executeDelegatecall(address target, bytes memory data) external onlyOwner {
        // Vulnerable: delegatecall can modify contract state
        (bool success, ) = target.delegatecall(data);
        require(success, "Delegatecall failed");
    }

    // Fallback function to receive ETH
    receive() external payable {}
}