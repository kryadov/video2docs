def test_app_starts_and_login_page():
    # Import here to avoid heavy imports if this test is skipped
    from src.webapp import create_app

    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        # Not logged in: index should redirect to login
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 303, 307, 308)
        # location header might be absolute or relative; just check it contains '/login'
        location = resp.headers.get("Location", "")
        assert "/login" in location

        # Login page should be accessible
        resp_login = client.get("/login")
        assert resp_login.status_code == 200
        # Basic sanity that it's HTML
        assert "text/html" in resp_login.headers.get("Content-Type", "")
