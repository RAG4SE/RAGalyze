def find_safe_utf8_boundary(data: bytes, position: int) -> int:
    """
    Find a safe UTF-8 character boundary near the given position.
    
    This function ensures we don't split in the middle of a multi-byte UTF-8 character.
    It looks backward from the position to find the start of a complete character.
    
    Args:
        data: The byte data
        position: The desired position
        
    Returns:
        A safe position that starts a complete UTF-8 character
    """
    if position >= len(data):
        return len(data)
    
    if position <= 0:
        return 0
    
    # Look backward up to 4 bytes to find a valid UTF-8 start
    for i in range(min(4, position + 1)):
        check_pos = position - i
        if check_pos < 0:
            break
            
        byte = data[check_pos]
        
        # Check if this is a valid UTF-8 start byte
        if byte <= 0x7F:  # ASCII (single byte)
            return check_pos
        elif (0xC2 <= byte <= 0xDF) or (0xE0 <= byte <= 0xEF) or (0xF0 <= byte <= 0xF7):
            try:
                if 0xC2 <= byte <= 0xDF:
                    end = check_pos + 2
                elif 0xE0 <= byte <= 0xEF:
                    end = check_pos + 3
                else:  # 0xF0 <= byte <= 0xF7
                    end = check_pos + 4
                end = min(end, len(data))
                test_slice = data[check_pos:end]
                test_slice.decode('utf-8', errors='strict')
                return check_pos
            except UnicodeDecodeError:
                continue
    
    return 0

if __name__ == "__main__":
    test_data = """
BOOST_AUTO_TEST_CASE(immediate_dominator_2)
{
        //    ┌────►A──────┐
        //    │     │      ▼
        //    │ B◄──┘   ┌──D──┐
        //    │ │       │     │
        //    │ ▼       ▼     ▼
        //    └─C◄───┐  E     F
        //      │    │  │     │
        //      └───►G◄─┴─────┘"""
    test_data = test_data.encode('utf-8')
    
    test_data_truncated = """
BOOST_AUTO_TEST_CASE(immediate_dominator_2)
{
        //    ┌────►A──────┐
        //    │     │      ▼
        //    │ B◄──┘   ┌──D──┐
        //    │ │       │     │
        //    │ ▼       ▼     ▼
        //    └─C◄───┐  E     F
        //      │    │  │     │
        //      └───►G◄─┴──"""
    test_data_truncated = test_data_truncated.encode('utf-8')
    
    print(
        len(test_data_truncated)
    )
    print(test_data[:find_safe_utf8_boundary(test_data, 404)].decode('utf-8'))