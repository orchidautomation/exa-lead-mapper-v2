import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class USAddressParser:
    """
    Utility class for parsing US addresses into components.
    
    Handles various US address formats and extracts:
    - Street address (number + street name)
    - City
    - State (2-letter code)
    - ZIP code (5 or 9 digit)
    """
    
    # Common US state abbreviations and variations
    US_STATES = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC'  # District of Columbia
    }
    
    # Street type abbreviations and variations
    STREET_TYPES = {
        'ST', 'STREET', 'AVE', 'AVENUE', 'RD', 'ROAD', 'BLVD', 'BOULEVARD',
        'LN', 'LANE', 'DR', 'DRIVE', 'CT', 'COURT', 'CIR', 'CIRCLE',
        'PL', 'PLACE', 'WAY', 'PKWY', 'PARKWAY', 'TER', 'TERRACE',
        'SQ', 'SQUARE', 'LOOP', 'PATH', 'TRAIL', 'HWY', 'HIGHWAY'
    }
    
    def __init__(self):
        """Initialize the address parser with compiled regex patterns."""
        # ZIP code pattern: 5 digits optionally followed by hyphen and 4 digits
        self.zip_pattern = re.compile(r'\b(\d{5}(?:-\d{4})?)\b')
        
        # State pattern: 2 letter state code
        self.state_pattern = re.compile(r'\b([A-Z]{2})\b')
        
        # Street number pattern: starts with digits
        self.street_number_pattern = re.compile(r'^(\d+\w?)\s+')
        
    def parse_address(self, address: str) -> Dict[str, Optional[str]]:
        """
        Parse a US address into components.
        
        Args:
            address: Full address string (e.g., "123 Main St, Springfield, IL 62701")
            
        Returns:
            Dictionary with parsed components:
            - street_address: Street number and name
            - city: City name
            - state: 2-letter state code
            - zip_code: ZIP code (5 or 9 digit)
        """
        result = {
            'street_address': None,
            'city': None,
            'state': None,
            'zip_code': None
        }
        
        if not address or not isinstance(address, str):
            return result
            
        # Clean the address
        address = address.strip()
        
        # Remove "United States" suffix if present
        address = re.sub(r',?\s*(United States|USA|US)$', '', address, flags=re.IGNORECASE)
        
        try:
            # Extract ZIP code first
            zip_match = self.zip_pattern.search(address)
            if zip_match:
                result['zip_code'] = zip_match.group(1)
                # Remove ZIP from address for further parsing
                address = address[:zip_match.start()].strip().rstrip(',')
            
            # Split by commas to get address components
            parts = [part.strip() for part in address.split(',')]
            
            if len(parts) >= 3:
                # Format: "123 Main St, Springfield, IL"
                street_part = parts[0]
                city_part = parts[1]
                state_part = parts[2]
                
                result['street_address'] = street_part
                result['city'] = city_part
                
                # Extract state from the state part
                state_match = self.state_pattern.search(state_part.upper())
                if state_match and state_match.group(1) in self.US_STATES:
                    result['state'] = state_match.group(1)
                    
            elif len(parts) == 2:
                # Format: "123 Main St, Springfield IL" or "Springfield, IL"
                first_part = parts[0]
                second_part = parts[1]
                
                # Check if first part has street number
                if self.street_number_pattern.match(first_part):
                    result['street_address'] = first_part
                    
                    # Second part should contain city and state
                    state_match = self.state_pattern.search(second_part.upper())
                    if state_match and state_match.group(1) in self.US_STATES:
                        result['state'] = state_match.group(1)
                        # City is everything before the state
                        city_text = second_part[:state_match.start()].strip()
                        if city_text:
                            result['city'] = city_text
                else:
                    # Assume first part is city, second is state
                    result['city'] = first_part
                    state_match = self.state_pattern.search(second_part.upper())
                    if state_match and state_match.group(1) in self.US_STATES:
                        result['state'] = state_match.group(1)
                        
            elif len(parts) == 1:
                # Single part - try to extract what we can
                single_part = parts[0]
                
                # Check for state
                state_match = self.state_pattern.search(single_part.upper())
                if state_match and state_match.group(1) in self.US_STATES:
                    result['state'] = state_match.group(1)
                    
                    # Check for street number at the beginning
                    if self.street_number_pattern.match(single_part):
                        # Likely contains street address
                        result['street_address'] = single_part
                    else:
                        # Likely just city and state
                        city_text = single_part[:state_match.start()].strip()
                        if city_text:
                            result['city'] = city_text
            
            # Validate and clean results
            for key, value in result.items():
                if value:
                    result[key] = value.strip()
                    if not result[key]:
                        result[key] = None
                        
        except Exception as e:
            logger.error(f"Error parsing address '{address}': {str(e)}")
            
        return result
    
    def extract_street_address(self, address: str) -> Optional[str]:
        """
        Extract just the street address component.
        
        Args:
            address: Full address string
            
        Returns:
            Street address part or None
        """
        parsed = self.parse_address(address)
        return parsed.get('street_address')
    
    def extract_zip_code(self, address: str) -> Optional[str]:
        """
        Extract just the ZIP code.
        
        Args:
            address: Full address string
            
        Returns:
            ZIP code or None
        """
        parsed = self.parse_address(address)
        return parsed.get('zip_code')
    
    def extract_state_code(self, address: str) -> Optional[str]:
        """
        Extract just the state code.
        
        Args:
            address: Full address string
            
        Returns:
            2-letter state code or None
        """
        parsed = self.parse_address(address)
        return parsed.get('state')


# Global instance for easy importing
us_address_parser = USAddressParser()