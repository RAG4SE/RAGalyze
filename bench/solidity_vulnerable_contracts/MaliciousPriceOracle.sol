// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Malicious Price Oracle
 * @dev A price oracle with manipulation vulnerabilities
 * Contains: Price manipulation, front-running, access control issues
 */

import "./VulnerableDeFiProtocol.sol";

interface IPriceFeed {
    function latestAnswer() external view returns (int256);
    function latestTimestamp() external view returns (uint256);
}

contract MaliciousPriceOracle is IPriceOracle {
    mapping(address => uint256) public prices;
    mapping(address => uint256) public lastUpdate;
    mapping(address => bool) public authorizedUpdaters;

    address public owner;
    uint256 public constant PRICE_PRECISION = 1e18;

    event PriceUpdated(address indexed token, uint256 oldPrice, uint256 newPrice, address updater);
    event UpdaterAdded(address indexed updater);
    event UpdaterRemoved(address indexed updater);

    modifier onlyAuthorized() {
        require(authorizedUpdaters[msg.sender], "Not authorized");
        _;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedUpdaters[msg.sender] = true;
    }

    /**
     * @dev VULNERABILITY: Price manipulation with no validation
     */
    function updatePrice(address token, uint256 newPrice) external override onlyAuthorized {
        uint256 oldPrice = prices[token];

        // Vulnerable: no price validation, no circuit breakers
        prices[token] = newPrice;
        lastUpdate[token] = block.timestamp;

        emit PriceUpdated(token, oldPrice, newPrice, msg.sender);
    }

    /**
     * @dev VULNERABILITY: Mass price update with single transaction
     */
    function updateMultiplePrices(address[] memory tokens, uint256[] memory newPrices) external onlyAuthorized {
        require(tokens.length == newPrices.length, "Arrays length mismatch");

        // Vulnerable: atomic update can be exploited
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 oldPrice = prices[tokens[i]];
            prices[tokens[i]] = newPrices[i];
            lastUpdate[tokens[i]] = block.timestamp;
            emit PriceUpdated(tokens[i], oldPrice, newPrices[i], msg.sender);
        }
    }

    /**
     * @dev VULNERABILITY: Front-running opportunity
     */
    function updatePriceWithDelay(address token, uint256 newPrice, uint256 delay) external onlyAuthorized {
        // Vulnerable: delay can be used for front-running
        if (block.timestamp >= lastUpdate[token] + delay) {
            uint256 oldPrice = prices[token];
            prices[token] = newPrice;
            lastUpdate[token] = block.timestamp;
            emit PriceUpdated(token, oldPrice, newPrice, msg.sender);
        }
    }

    /**
     * @dev VULNERABILITY: Flash loan attack vulnerability
     */
    function flashLoanPriceManipulation(address token, uint256 manipulatedPrice) external {
        // Vulnerable: no check for flash loan origins
        // An attacker can borrow tokens, manipulate price, and repay in same transaction
        uint256 oldPrice = prices[token];
        prices[token] = manipulatedPrice;
        lastUpdate[token] = block.timestamp;
        emit PriceUpdated(token, oldPrice, manipulatedPrice, msg.sender);

        // Execute callback - vulnerable to reentrancy
        (bool success, ) = msg.sender.call(abi.encodeWithSignature("priceManipulationCallback()"));
        require(success, "Price manipulation callback failed");
    }

    /**
     * @dev VULNERABILITY: Unauthorized access to price feeds
     */
    function setPriceFromFeed(address token, address priceFeed) external {
        // Vulnerable: no access control
        int256 feedPrice = IPriceFeed(priceFeed).latestAnswer();
        if (feedPrice > 0) {
            uint256 oldPrice = prices[token];
            prices[token] = uint256(feedPrice) * PRICE_PRECISION;
            lastUpdate[token] = block.timestamp;
            emit PriceUpdated(token, oldPrice, prices[token], msg.sender);
        }
    }

    /**
     * @dev VULNERABILITY: Timestamp manipulation
     */
    function updatePriceWithTimestamp(address token, uint256 newPrice, uint256 timestamp) external onlyAuthorized {
        // Vulnerable: can set future/past timestamps
        prices[token] = newPrice;
        lastUpdate[token] = timestamp;
        emit PriceUpdated(token, prices[token], newPrice, msg.sender);
    }

    /**
     * @dev VULNERABILITY: Price oracle manipulation through constructor
     */
    function initializePrice(address token, uint256 initialPrice) external onlyOwner {
        // Vulnerable: can be called multiple times to reset prices
        prices[token] = initialPrice;
        lastUpdate[token] = block.timestamp;
        emit PriceUpdated(token, 0, initialPrice, msg.sender);
    }

    function getPrice(address token) external view override returns (uint256) {
        // Vulnerable: returns 0 if price not set
        return prices[token];
    }

    function getLastUpdate(address token) external view returns (uint256) {
        return lastUpdate[token];
    }

    function addUpdater(address updater) external onlyOwner {
        authorizedUpdaters[updater] = true;
        emit UpdaterAdded(updater);
    }

    function removeUpdater(address updater) external onlyOwner {
        authorizedUpdaters[updater] = false;
        emit UpdaterRemoved(updater);
    }

    /**
     * @dev VULNERABILITY: Emergency function with no validation
     */
    function emergencyUpdate(address token, uint256 newPrice) external onlyOwner {
        // Vulnerable: can be used to manipulate prices without validation
        prices[token] = newPrice;
        lastUpdate[token] = block.timestamp;
        emit PriceUpdated(token, prices[token], newPrice, msg.sender);
    }

    /**
     * @dev VULNERABILITY: Batch manipulation
     */
    function manipulateMultipleTokens(address[] memory tokens, uint256[] memory pricesToSet) external onlyOwner {
        require(tokens.length == pricesToSet.length, "Arrays length mismatch");

        // Vulnerable: can manipulate multiple token prices simultaneously
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 oldPrice = prices[tokens[i]];
            prices[tokens[i]] = pricesToSet[i];
            lastUpdate[tokens[i]] = block.timestamp;
            emit PriceUpdated(tokens[i], oldPrice, pricesToSet[i], msg.sender);
        }
    }

    /**
     * @dev VULNERABILITY: Price manipulation with minimal delay
     */
    function quickUpdate(address token, uint256 newPrice) external onlyAuthorized {
        // Vulnerable: no rate limiting
        require(block.timestamp >= lastUpdate[token] + 1 seconds, "Too soon");

        uint256 oldPrice = prices[token];
        prices[token] = newPrice;
        lastUpdate[token] = block.timestamp;
        emit PriceUpdated(token, oldPrice, newPrice, msg.sender);
    }
}