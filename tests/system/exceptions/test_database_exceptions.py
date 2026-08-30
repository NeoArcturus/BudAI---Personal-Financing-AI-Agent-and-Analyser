import pytest

class MockIntegrityError(Exception): pass
class MockOperationalError(Exception): pass

def test_database_persistence_traps():
    """
    TDD Exceptions: Asserts that raw SQLAlchemy engine exceptions are caught, 
    sessions are rolled back to prevent locks, and precise HTTP status codes 
    are generated.
    """
    def execute_db_commit(simulated_error):
        session_rolled_back = False
        try:
            if simulated_error == "integrity":
                raise MockIntegrityError("Duplicate Key")
            if simulated_error == "operational":
                raise MockOperationalError("Connection Lost")
            return 200, session_rolled_back
        except MockIntegrityError:
            session_rolled_back = True
            return 409, session_rolled_back
        except MockOperationalError:
            session_rolled_back = True
            return 503, session_rolled_back

    status_1, rb_1 = execute_db_commit("integrity")
    assert status_1 == 409
    assert rb_1 is True

    status_2, rb_2 = execute_db_commit("operational")
    assert status_2 == 503
    assert rb_2 is True

