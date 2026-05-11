import streamlit as st
from datetime import datetime
import pytz
from zoneinfo import ZoneInfo
import time

# Page configuration
st.set_page_config(
    page_title="🕐 Digital Clock - World Time Zones",
    page_icon="🕐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .clock-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        text-align: center;
    }
    
    .clock-time {
        font-size: 48px;
        font-weight: bold;
        color: #ffffff;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
    }
    
    .clock-timezone {
        font-size: 24px;
        color: #e0e0e0;
        margin: 5px 0;
    }
    
    .clock-location {
        font-size: 18px;
        color: #ffffff;
        opacity: 0.8;
        margin: 5px 0;
    }
    
    .clock-info {
        font-size: 14px;
        color: #ffffff;
        opacity: 0.7;
        margin-top: 15px;
    }
    
    .title-main {
        text-align: center;
        color: #667eea;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 12px;
        font-weight: bold;
    }
    
    .metric-value {
        color: #667eea;
        font-size: 18px;
        font-weight: bold;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-main">🕐 Digital Clock - World Time Zones</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time clock across different time zones</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Settings")

# Common time zones
common_timezones = {
    "🌍 World Major Cities": {
        "UTC (Coordinated Universal Time)": "UTC",
        "🇬🇧 London (GMT/BST)": "Europe/London",
        "🇪🇺 Paris (CET/CEST)": "Europe/Paris",
        "🇵🇹 Lisbon (WET/WEST)": "Europe/Lisbon",
        "🇪🇸 Madrid (CET/CEST)": "Europe/Madrid",
        "🇮🇹 Rome (CET/CEST)": "Europe/Rome",
        "🇬🇷 Athens (EET/EEST)": "Europe/Athens",
        "🇹🇷 Istanbul (EET)": "Europe/Istanbul",
        "🇸🇦 Dubai (GST)": "Asia/Dubai",
        "🇮🇳 India (IST)": "Asia/Kolkata",
        "🇹🇭 Bangkok (ICT)": "Asia/Bangkok",
        "🇸🇬 Singapore (SGT)": "Asia/Singapore",
        "🇭🇰 Hong Kong (HKT)": "Asia/Hong_Kong",
        "🇯🇵 Tokyo (JST)": "Asia/Tokyo",
        "🇦🇺 Sydney (AEDT/AEST)": "Australia/Sydney",
        "🇺🇸 New York (EST/EDT)": "America/New_York",
        "🇺🇸 Los Angeles (PST/PDT)": "America/Los_Angeles",
        "🇨🇦 Toronto (EST/EDT)": "America/Toronto",
        "🇲🇽 Mexico City (CST/CDT)": "America/Mexico_City",
        "🇧🇷 São Paulo (BRT/BRST)": "America/Sao_Paulo",
        "🇦🇷 Buenos Aires (ART)": "America/Argentina/Buenos_Aires",
        "🇿🇦 Johannesburg (SAST)": "Africa/Johannesburg",
    }
}

# Select display mode
display_mode = st.sidebar.radio(
    "Choose Display Mode",
    ["🎯 Selected Time Zones", "🌍 All Available Zones", "⏰ Analog Clock Simulation"]
)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (updates every 1 second)", value=True)

if auto_refresh:
    st.markdown(
        """
        <script>
            setTimeout(function() {
                window.location.reload();
            }, 1000);
        </script>
        """,
        unsafe_allow_html=True,
    )

# Main content area
if display_mode == "🎯 Selected Time Zones":
    st.sidebar.markdown("---")
    st.sidebar.title("📍 Select Time Zones")
    
    selected_zones = st.sidebar.multiselect(
        "Choose time zones to display:",
        options=list(common_timezones["🌍 World Major Cities"].keys()),
        default=[
            "UTC (Coordinated Universal Time)",
            "🇬🇧 London (GMT/BST)",
            "🇪🇺 Paris (CET/CEST)",
            "🇮🇳 India (IST)",
            "🇯🇵 Tokyo (JST)",
            "🇺🇸 New York (EST/EDT)",
            "🇺🇸 Los Angeles (PST/PDT)",
        ]
    )
    
    if not selected_zones:
        st.warning("⚠️ Please select at least one time zone")
    else:
        # Display selected clocks
        cols = st.columns(2)
        for idx, zone_label in enumerate(selected_zones):
            tz_name = common_timezones["🌍 World Major Cities"][zone_label]
            tz = pytz.timezone(tz_name)
            current_time = datetime.now(tz)
            
            with cols[idx % 2]:
                # Format time with different styles
                time_str = current_time.strftime("%H:%M:%S")
                date_str = current_time.strftime("%A, %B %d, %Y")
                offset = current_time.strftime("%z")
                
                # Extract location name
                location = zone_label.split()[-1].strip("()")
                
                st.markdown(
                    f"""
                    <div class="clock-container">
                        <div class="clock-location">{zone_label.split()[-1].rstrip(')')}</div>
                        <div class="clock-time">{time_str}</div>
                        <div class="clock-timezone">{tz_name}</div>
                        <div class="clock-info">
                            {date_str}<br>
                            UTC {offset[:3]}:{offset[3:]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

elif display_mode == "🌍 All Available Zones":
    st.subheader("🌐 All World Time Zones")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        search_query = st.text_input("🔍 Search time zones:", placeholder="e.g., London, Tokyo, New York")
    
    with col2:
        sort_by = st.selectbox("Sort by:", ["UTC Offset", "Alphabetical"])
    
    # Get all timezones
    all_zones = common_timezones["🌍 World Major Cities"]
    
    # Filter by search
    if search_query:
        filtered_zones = {k: v for k, v in all_zones.items() if search_query.lower() in k.lower()}
    else:
        filtered_zones = all_zones
    
    # Sort if requested
    if sort_by == "UTC Offset":
        # Sort by UTC offset
        sorted_zones = sorted(
            filtered_zones.items(),
            key=lambda x: datetime.now(pytz.timezone(x[1])).strftime("%z")
        )
        filtered_zones = dict(sorted_zones)
    
    # Display all zones in a grid
    cols = st.columns(3)
    
    for idx, (zone_label, tz_name) in enumerate(filtered_zones.items()):
        tz = pytz.timezone(tz_name)
        current_time = datetime.now(tz)
        
        time_str = current_time.strftime("%H:%M:%S")
        offset = current_time.strftime("%z")
        
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="clock-container" style="padding: 20px;">
                    <div class="clock-timezone" style="font-size: 16px;">{zone_label}</div>
                    <div class="clock-time" style="font-size: 32px;">{time_str}</div>
                    <div class="clock-info">
                        {tz_name}<br>
                        UTC {offset[:3]}:{offset[3:]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

else:  # Analog Clock Simulation
    st.subheader("⏰ Analog Clock Simulation")
    
    # Select timezone for analog clock
    selected_analog = st.selectbox(
        "Select time zone for analog clock:",
        options=list(common_timezones["🌍 World Major Cities"].keys()),
        index=0
    )
    
    tz_name = common_timezones["🌍 World Major Cities"][selected_analog]
    tz = pytz.timezone(tz_name)
    current_time = datetime.now(tz)
    
    # Extract time components
    hour = current_time.hour % 12
    minute = current_time.minute
    second = current_time.second
    
    # Calculate angles (in degrees, starting from 12 o'clock)
    second_angle = (second * 6) % 360
    minute_angle = (minute * 6 + second * 0.1) % 360
    hour_angle = (hour * 30 + minute * 0.5) % 360
    
    # Display digital info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">HOURS</div>
                <div class="metric-value">{current_time.strftime("%H")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MINUTES</div>
                <div class="metric-value">{current_time.strftime("%M")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">SECONDS</div>
                <div class="metric-value">{current_time.strftime("%S")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TIMEZONE</div>
                <div class="metric-value">{tz_name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Draw ASCII clock representation
    st.markdown("### 🕐 Clock Representation")
    
    # Create a simple text-based clock
    clock_display = f"""
    ╔═══════════════════════╗
    ║                       ║
    ║        🕐 CLOCK       ║
    ║                       ║
    ║   Hour: {'═' * int(hour * 2.5)}                ║
    ║   Minute: {'═' * int(minute * 0.4)}            ║
    ║   Second: {'═' * int(second * 0.4)}            ║
    ║                       ║
    ║   Time: {current_time.strftime('%H:%M:%S')}          ║
    ║   Date: {current_time.strftime('%Y-%m-%d')}      ║
    ║   Zone: {tz_name:<17} ║
    ║                       ║
    ╚═══════════════════════╝
    """
    
    st.code(clock_display, language="text")
    
    # Display angle information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hour Hand Angle", f"{hour_angle:.1f}°", delta="Rotates 30° per hour")
    
    with col2:
        st.metric("Minute Hand Angle", f"{minute_angle:.1f}°", delta="Rotates 6° per minute")
    
    with col3:
        st.metric("Second Hand Angle", f"{second_angle:.1f}°", delta="Rotates 6° per second")

# Additional information in expandable section
with st.expander("ℹ️ About Time Zones"):
    st.markdown("""
    ### Time Zone Information
    
    **UTC (Coordinated Universal Time)**
    - The primary time standard used worldwide
    - Reference point for all other time zones
    - Replaces the older GMT (Greenwich Mean Time)
    
    **Time Zone Abbreviations**
    - **EST/EDT:** Eastern Standard/Daylight Time (UTC-5/-4)
    - **CST/CDT:** Central Standard/Daylight Time (UTC-6/-5)
    - **MST/MDT:** Mountain Standard/Daylight Time (UTC-7/-6)
    - **PST/PDT:** Pacific Standard/Daylight Time (UTC-8/-7)
    - **GMT/BST:** Greenwich Mean Time / British Summer Time (UTC+0/+1)
    - **CET/CEST:** Central European Time / Summer Time (UTC+1/+2)
    - **IST:** Indian Standard Time (UTC+5:30)
    - **JST:** Japan Standard Time (UTC+9)
    - **AEDT/AEST:** Australian Eastern Daylight/Standard Time (UTC+11/+10)
    
    **Daylight Saving Time (DST)**
    - Many countries adjust their clocks forward in spring and backward in fall
    - Typically one hour adjustment
    - Different countries start/end DST on different dates
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🕐 Digital Clock - World Time Zones Display</p>
    <p>Built with Streamlit | Supports 24+ Major Time Zones</p>
</div>
""", unsafe_allow_html=True)
