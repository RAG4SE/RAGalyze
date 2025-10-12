// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Vulnerable DeFi Lending Protocol
 * @dev A vulnerable lending protocol with classic smart contract vulnerabilities
 * Contains: Reentrancy, Integer Overflow/Underflow, Unchecked External Calls, Access Control Issues
 */

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
}

interface IPriceOracle {
    function getPrice(address token) external view returns (uint256);
    function updatePrice(address token, uint256 newPrice) external;
}

contract VulnerableDeFiProtocol {
    struct Loan {
        address borrower;
        address collateralToken;
        address borrowToken;
        uint256 collateralAmount;
        uint256 borrowAmount;
        uint256 timestamp;
        bool isActive;
    }

    mapping(address => mapping(address => uint256)) public deposits;
    mapping(address => mapping(address => uint256)) public borrowBalances;
    mapping(uint256 => Loan) public loans;
    mapping(address => uint256[]) public userLoans;

    address public owner;
    IPriceOracle public priceOracle;
    uint256 public loanCounter;
    uint256 public interestRate = 500; // 5% in basis points
    uint256 public collateralRatio = 150; // 150%

    event Deposit(address indexed user, address indexed token, uint256 amount);
    event Withdraw(address indexed user, address indexed token, uint256 amount);
    event Borrow(address indexed user, uint256 indexed loanId, address collateralToken, address borrowToken, uint256 collateralAmount, uint256 borrowAmount);
    event Repay(uint256 indexed loanId, uint256 amount);
    event Liquidate(uint256 indexed loanId, address liquidator);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor(address _priceOracle) {
        owner = msg.sender;
        priceOracle = IPriceOracle(_priceOracle);
    }

    /**
     * @dev VULNERABILITY: Reentrancy in withdraw function
     * User can re-enter the withdraw function by implementing a malicious receive() function
     */
    function withdraw(address token, uint256 amount) external {
        require(deposits[msg.sender][token] >= amount, "Insufficient balance");

        // Vulnerable: state change after external call
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        deposits[msg.sender][token] -= amount; // State change AFTER external call
        emit Withdraw(msg.sender, token, amount);
    }

    /**
     * @dev VULNERABILITY: Integer overflow in deposit calculation
     */
    function deposit(address token, uint256 amount) external {
        require(amount > 0, "Amount must be > 0");

        // Vulnerable: potential overflow
        deposits[msg.sender][token] += amount;

        // Vulnerable: unchecked external call
        IERC20(token).transferFrom(msg.sender, address(this), amount);

        emit Deposit(msg.sender, token, amount);
    }

    /**
     * @dev VULNERABILITY: Unchecked low-level call in borrow function
     */
    function borrow(
        address collateralToken,
        address borrowToken,
        uint256 collateralAmount,
        uint256 borrowAmount
    ) external returns (uint256) {
        require(collateralAmount > 0 && borrowAmount > 0, "Amounts must be > 0");

        // Get token prices
        uint256 collateralPrice = priceOracle.getPrice(collateralToken);
        uint256 borrowPrice = priceOracle.getPrice(borrowToken);

        // Calculate collateral value
        uint256 collateralValue = (collateralAmount * collateralPrice) / 1e18;

        // Calculate required collateral
        uint256 requiredCollateral = (borrowAmount * borrowPrice * collateralRatio) / 100 / 1e18;

        require(collateralValue >= requiredCollateral, "Insufficient collateral");

        // Transfer collateral
        // Vulnerable: unchecked external call
        (bool success, ) = address(this).call(abi.encodeWithSignature("transferFrom(address,address,uint256)", msg.sender, address(this), collateralAmount));
        require(success, "Collateral transfer failed");

        // Transfer borrowed tokens
        // Vulnerable: unchecked external call
        (success, ) = address(this).call(abi.encodeWithSignature("transfer(address,uint256)", msg.sender, borrowAmount));
        require(success, "Borrow transfer failed");

        // Create loan
        loanCounter++;
        loans[loanCounter] = Loan({
            borrower: msg.sender,
            collateralToken: collateralToken,
            borrowToken: borrowToken,
            collateralAmount: collateralAmount,
            borrowAmount: borrowAmount,
            timestamp: block.timestamp,
            isActive: true
        });

        userLoans[msg.sender].push(loanCounter);
        borrowBalances[msg.sender][borrowToken] += borrowAmount;

        emit Borrow(msg.sender, loanCounter, collateralToken, borrowToken, collateralAmount, borrowAmount);
        return loanCounter;
    }

    /**
     * @dev VULNERABILITY: Reentrancy in repay function
     */
    function repay(uint256 loanId, uint256 amount) external {
        Loan storage loan = loans[loanId];
        require(loan.isActive, "Loan not active");
        require(loan.borrower == msg.sender, "Not borrower");

        // Calculate interest
        uint256 interest = calculateInterest(loan.borrowAmount, loan.timestamp);
        uint256 totalRepay = loan.borrowAmount + interest;

        require(amount <= totalRepay, "Amount exceeds total");

        // Vulnerable: external call before state update
        IERC20(loan.borrowToken).transferFrom(msg.sender, address(this), amount);

        if (amount == totalRepay) {
            // Return collateral - vulnerable to reentrancy
            IERC20(loan.collateralToken).transfer(msg.sender, loan.collateralAmount);
            loan.isActive = false;
        }

        borrowBalances[msg.sender][loan.borrowToken] -= amount;
        emit Repay(loanId, amount);
    }

    /**
     * @dev VULNERABILITY: Access control - anyone can update price oracle
     */
    function updatePrice(address token, uint256 newPrice) external {
        // Vulnerable: no access control
        priceOracle.updatePrice(token, newPrice);
    }

    /**
     * @dev VULNERABILITY: Integer underflow in liquidation calculation
     */
    function liquidate(uint256 loanId) external {
        Loan storage loan = loans[loanId];
        require(loan.isActive, "Loan not active");

        uint256 collateralPrice = priceOracle.getPrice(loan.collateralToken);
        uint256 borrowPrice = priceOracle.getPrice(loan.borrowToken);

        // Calculate current collateral value
        uint256 currentCollateralValue = (loan.collateralAmount * collateralPrice) / 1e18;
        uint256 borrowValue = (loan.borrowAmount * borrowPrice) / 1e18;

        // Check if loan is undercollateralized
        require(currentCollateralValue < borrowValue, "Loan not undercollateralized");

        // Vulnerable: potential underflow
        uint256 liquidationBonus = borrowValue * 110 / 100; // 10% bonus
        uint256 totalPayout = borrowValue + liquidationBonus;

        require(totalPayout <= currentCollateralValue, "Insufficient collateral");

        // Transfer liquidation amount
        // Vulnerable: unchecked external call
        IERC20(loan.borrowToken).transferFrom(msg.sender, address(this), loan.borrowAmount);

        // Transfer collateral to liquidator
        // Vulnerable: unchecked external call
        IERC20(loan.collateralToken).transfer(msg.sender, loan.collateralAmount);

        loan.isActive = false;
        emit Liquidate(loanId, msg.sender);
    }

    /**
     * @dev VULNERABILITY: Flash loan attack vulnerability
     */
    function flashLoan(uint256 amount, address token, address borrower) external {
        require(amount > 0, "Amount must be > 0");
        require(IERC20(token).balanceOf(address(this)) >= amount, "Insufficient liquidity");

        // Transfer tokens to borrower
        IERC20(token).transfer(borrower, amount);

        // Execute callback - vulnerable to reentrancy
        (bool success, ) = borrower.call(abi.encodeWithSignature("flashLoanCallback(uint256,address)", amount, token));
        require(success, "Flash loan callback failed");

        // Check if tokens were returned
        require(IERC20(token).balanceOf(address(this)) >= amount, "Flash loan not repaid");
    }

    /**
     * @dev VULNERABILITY: Unchecked return value in emergency function
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        // Vulnerable: unchecked return value
        IERC20(token).transfer(owner, amount);
    }

    function calculateInterest(uint256 principal, uint256 startTime) internal view returns (uint256) {
        uint256 timeElapsed = block.timestamp - startTime;
        return (principal * interestRate * timeElapsed) / (365 days * 10000);
    }

    /**
     * @dev VULNERABILITY: DoS with unbounded loop
     */
    function getUserLoans(address user) external view returns (Loan[] memory) {
        uint256[] memory loanIds = userLoans[user];
        Loan[] memory userLoanData = new Loan[](loanIds.length);

        // Vulnerable: unbounded loop
        for (uint256 i = 0; i < loanIds.length; i++) {
            userLoanData[i] = loans[loanIds[i]];
        }

        return userLoanData;
    }

    /**
     * @dev VULNERABILITY: Front-running vulnerability
     */
    function updateInterestRate(uint256 newRate) external onlyOwner {
        // Vulnerable: front-running possible
        interestRate = newRate;
    }

    /**
     * @dev VULNERABILITY: Arbitrary call through delegatecall
     */
    function executeDelegatecall(address target, bytes memory data) external onlyOwner {
        // Vulnerable: delegatecall can modify contract state
        (bool success, ) = target.delegatecall(data);
        require(success, "Delegatecall failed");
    }

    /**
     * @dev VULNERABILITY: Storage collision with upgradeable proxy pattern
     */
    function upgradeTo(address newImplementation) external onlyOwner {
        // This would be vulnerable in a proxy setup
        // For demonstration, we'll just emit an event
        // In a real proxy, this could lead to storage collision
    }

    // Fallback function to receive ETH
    receive() external payable {}
}