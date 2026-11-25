"""Optional access-code gate, required on the shared cloud deployment."""
import hmac
import os
import streamlit as st


def require_access():
    code = os.environ.get("LAB_ACCESS_CODE", "")
    if not code:
        try:
            code = str(st.secrets.get("LAB_ACCESS_CODE", ""))
        except FileNotFoundError:
            pass
    if not code:
        if os.environ.get("LAB_REQUIRE_AUTH") == "1":
            st.error("The lab access secret is missing. Configure vector-lab-access in Modal.")
            st.stop()
        return
    if st.session_state.get("authenticated"):
        return
    st.title("Vector Autonomy Lab")
    st.caption("Enter your lab access code to open the simulation workspace.")
    with st.form("lab_sign_in"):
        supplied = st.text_input("Lab access code", type="password")
        submit = st.form_submit_button("Open workspace", type="primary")
    if submit:
        if hmac.compare_digest(supplied.encode(), code.encode()):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Access code not recognized.")
    st.stop()
