echo Running $0

no_wrist=$1
digit_i=$2
digit_ii=$3
digit_iii=$4
digit_iv=$5
digit_v=$6

echo No Wrist: $no_wrist
echo Digit I: $digit_i
echo Digit II: $digit_ii
echo Digit III: $digit_iii
echo Digit IV: $digit_iv
echo Digit V: $digit_v


rosrun xacro xacro darm_hand.xml.xacro -o darm_hand.xml no_wrist:=$no_wrist digit_i:=$digit_i digit_ii:=$digit_ii digit_iii:=$digit_iii digit_iv:=$digit_iv digit_v:=$digit_v
rosrun xacro xacro contact_exclusions.xml.xacro -o contact_exclusions.xml digit_i:=$digit_i digit_ii:=$digit_ii digit_iii:=$digit_iii digit_iv:=$digit_iv digit_v:=$digit_v
rosrun xacro xacro tendon.xml.xacro -o tendon.xml no_wrist:=$no_wrist digit_i:=$digit_i digit_ii:=$digit_ii digit_iii:=$digit_iii digit_iv:=$digit_iv digit_v:=$digit_v
rosrun xacro xacro actuator.xml.xacro -o actuator.xml no_wrist:=$no_wrist digit_i:=$digit_i digit_ii:=$digit_ii digit_iii:=$digit_iii digit_iv:=$digit_iv digit_v:=$digit_v