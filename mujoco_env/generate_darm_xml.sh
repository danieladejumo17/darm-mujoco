echo Running $0

single_finger=$1
no_wrist=$2
echo Single Finger: $single_finger
echo No Wrist: $no_wrist


rosrun xacro xacro darm_hand.xml.xacro -o darm_hand.xml single_finger:=$single_finger no_wrist:=$no_wrist
rosrun xacro xacro contact_exclusions.xml.xacro -o contact_exclusions.xml single_finger:=$single_finger
rosrun xacro xacro tendon.xml.xacro -o tendon.xml single_finger:=$single_finger no_wrist:=$no_wrist
rosrun xacro xacro actuator.xml.xacro -o actuator.xml single_finger:=$single_finger no_wrist:=$no_wrist