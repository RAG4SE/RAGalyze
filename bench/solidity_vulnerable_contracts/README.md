# Solidity Vulnerable Contracts Benchmark

This benchmark contains a set of Solidity smart contracts with intentionally injected vulnerabilities to test function call chain extraction and vulnerability detection capabilities.

## Contract Overview

### 1. VulnerableDeFiProtocol.sol
A complex DeFi lending protocol with multiple vulnerability patterns:

**Vulnerabilities:**
- **Reentrancy**: `withdraw()` and `repay()` functions vulnerable to reentrancy attacks
- **Integer Overflow/Underflow**: `deposit()` and `liquidate()` functions
- **Unchecked External Calls**: Multiple functions use low-level calls without validation
- **Access Control Issues**: `updatePrice()` function lacks proper access controls
- **Flash Loan Attacks**: `flashLoan()` function vulnerable to manipulation
- **DoS with Unbounded Loops**: `getUserLoans()` function
- **Front-running**: `updateInterestRate()` function
- **Delegatecall Vulnerability**: `executeDelegatecall()` function

**Complex Function Calls:**
- Nested function calls with external dependencies
- Template/generic function patterns
- Complex inheritance chains
- Multi-step transaction flows

### 2. MaliciousPriceOracle.sol
A price oracle contract with manipulation vulnerabilities:

**Vulnerabilities:**
- **Price Manipulation**: No validation on price updates
- **Front-running**: Timestamp-based manipulation opportunities
- **Flash Loan Attacks**: `flashLoanPriceManipulation()` function
- **Access Control Issues**: `setPriceFromFeed()` lacks proper controls
- **Batch Manipulation**: `manipulateMultipleTokens()` function
- **Rate Limiting Bypass**: `quickUpdate()` function

**Complex Function Calls:**
- External price feed integrations
- Batch update operations
- Callback mechanisms
- Time-based operations

### 3. ComplexToken.sol
An ERC20-like token with complex vulnerability patterns:

**Vulnerabilities:**
- **Reentrancy**: `transfer()` and `transferFrom()` functions
- **Integer Arithmetic Issues**: Multiple functions with overflow/underflow
- **Flash Loan Vulnerability**: `flashLoan()` function
- **Access Control Bypass**: `emergencyMint()` function
- **DoS with Loops**: `airdrop()` and `multiTransfer()` functions
- **Self-destruct Vulnerability**: `emergencyWithdraw()` function
- **Delegatecall Risk**: `executeDelegatecall()` function

**Complex Function Calls:**
- Multi-token operations
- Fee calculations
- Time-based restrictions
- Batch operations

### 4. AttackContract.sol
A contract demonstrating various attack patterns:

**Attack Types:**
- **Reentrancy Attacks**: `launchReentrancyAttack()`
- **Flash Loan Attacks**: `launchFlashLoanAttack()`
- **Price Manipulation**: `launchPriceManipulationAttack()`
- **Front-running**: `launchFrontRunningAttack()`
- **Complex Multi-vector**: `launchComplexAttack()`
- **DoS Attacks**: `launchDoSAttack()`
- **Fund Draining**: `drainFunds()`

**Complex Function Calls:**
- Callback mechanisms
- Multi-step attack sequences
- Cross-contract interactions
- Complex state manipulation

## Function Call Chain Complexity

These contracts are designed to test function call chain extraction with:

1. **Deep Call Stacks**: Functions that call other functions recursively
2. **Cross-Contract Calls**: Multiple contracts interacting with each other
3. **Conditional Logic**: Complex branching based on state
4. **External Dependencies**: Calls to oracles and external contracts
5. **Event-Driven Patterns**: Functions triggered by events
6. **Fallback/Receive Functions**: Special function types
7. **Template/Generic Patterns**: Functions with type parameters
8. **Inheritance Hierarchies**: Complex parent-child relationships

## Vulnerability Categories

### Financial Vulnerabilities
- Reentrancy
- Integer overflow/underflow
- Flash loan attacks
- Price manipulation

### Access Control Vulnerabilities
- Missing access controls
- Privilege escalation
- Function visibility issues

### Denial of Service Vulnerabilities
- Unbounded loops
- Gas limit exhaustion
- Storage exhaustion

### Logic Flaws
- Front-running
- Timestamp manipulation
- Race conditions

### Low-Level Vulnerabilities
- Unchecked external calls
- Delegatecall misuse
- Self-destruct abuse

## Testing Scenarios

These contracts provide rich testing scenarios for:

1. **Function Call Chain Extraction**: Complex nested calls and dependencies
2. **Vulnerability Detection**: Multiple vulnerability patterns to identify
3. **Code Analysis**: Static and dynamic analysis testing
4. **Security Auditing**: Comprehensive security assessment
5. **Penetration Testing**: Real-world attack simulation

## Usage

```solidity
// Deploy contracts in order
VulnerableDeFiProtocol protocol = new VulnerableDeFiProtocol(oracleAddress);
MaliciousPriceOracle oracle = new MaliciousPriceOracle();
ComplexToken token = new ComplexToken("Test Token", "TST", 18, 1000000);
AttackContract attacker = new AttackContract(address(protocol), address(oracle), address(token));

// Launch various attacks
attacker.launchReentrancyAttack(address(token), 1000);
attacker.launchFlashLoanAttack(address(token), 5000);
attacker.launchComplexAttack(address(token), 2000);
```

## Security Warning

⚠️ **These contracts contain intentional vulnerabilities and should NEVER be deployed to mainnet or used in production environments. They are designed for testing and educational purposes only.**