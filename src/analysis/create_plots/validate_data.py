from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Stores the result of a single validation test."""
    input_text: str
    expected: Optional[int]
    actual: Optional[int]
    passed: bool

    def format_result(self) -> str:
        """Creates a formatted string representation of the test result."""
        status = "✓" if self.passed else "✗"
        expected_str = str(self.expected) if self.expected is not None else "None"
        actual_str = str(self.actual) if self.actual is not None else "None"
        return f"{status} Input: {self.input_text:<20} Expected: {expected_str:<4} Got: {actual_str}"


class AnswerMapper:
    """Maps answer strings to valid multiple choice positions."""

    POSITION_MAPPINGS = {
        'greek': "αβγδεζηθικ",
        'keyboard': "!@#$%^₪*)(",
        'capitals': "ABCDEFGHIJ",
        'lowercase': "abcdefghij",
        'numbers': [str(i + 1) for i in range(10)],
        'roman': ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    }

    VALID_POSITIONS = {1, 2, 3, 4}  # Valid multiple choice positions

    @classmethod
    def get_position(cls, answer: str) -> Optional[int]:
        """
        Maps an answer string to a valid multiple choice position.

        Args:
            answer: String containing the answer prefix (e.g., "A.", "1)", "α.")

        Returns:
            int: Position (1-4) if valid
            None: If mapping is invalid or position is out of range
        """
        if not answer or not isinstance(answer, str):
            return None

        try:
            prefix = answer.split('.')[0].strip()

            for mapping in cls.POSITION_MAPPINGS.values():
                if prefix in mapping:
                    position = mapping.index(prefix) + 1
                    return position if position in cls.VALID_POSITIONS else None

            return None

        except (AttributeError, IndexError):
            return None


def validate_answer_mapper() -> List[ValidationResult]:
    """
    Validates the AnswerMapper with comprehensive test cases.

    Returns:
        List[ValidationResult]: Results of all validation tests
    """
    test_cases = [
        # Valid cases (positions 1-4)
        ("A. First answer", 1),
        ("B. Second answer", 2),
        ("γ. Third answer", 3),
        ("IV. Fourth answer", 4),
        ("4. Fourth numeric", 4),
        ("!. First keyboard", 1),

        # Invalid cases (positions > 4)
        ("E. Fifth answer", None),
        ("ε. Fifth greek", None),
        ("VI. Sixth roman", None),
        ("6. Sixth numeric", None),

        # Edge cases
        ("", None),
        (None, None),
        ("Invalid input", None),
        ("10. Too high", None),
        ("X. Invalid roman", None)
    ]

    results = []
    mapper = AnswerMapper()

    for input_text, expected in test_cases:
        actual = mapper.get_position(input_text)
        results.append(ValidationResult(
            input_text=str(input_text),
            expected=expected,
            actual=actual,
            passed=actual == expected
        ))

    return results


def main():
    """Runs answer mapper validation and displays results."""
    print("Answer Mapper Validation")
    print("=" * 60)

    results = validate_answer_mapper()

    # Display results grouped by status
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print(f"\nPassed Tests: {len(passed)}/{len(results)}")
    for result in passed:
        print(result.format_result())

    if failed:
        print(f"\nFailed Tests: {len(failed)}/{len(results)}")
        for result in failed:
            print(result.format_result())
    else:
        print("\nAll tests passed successfully!")


if __name__ == "__main__":
    main()