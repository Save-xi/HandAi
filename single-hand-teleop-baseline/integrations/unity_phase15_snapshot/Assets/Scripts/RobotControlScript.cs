using UnityEngine;
using ExchangeSerialization;
using System.Collections.Concurrent;
using FlatBuffers;
using DriverSVH;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using WebSocketSharp;
using LitJson;
using System.Collections.Generic;

public class RobotControlScript : MonoBehaviour
{
    [SerializeField]
    private ArticulationBody[] virtualSVH;
    [SerializeField]
    private bool enableBaselineUdpPreview = true;
    [SerializeField]
    private int baselineUdpListenPort = 18080;
    private bool applyBaselinePreviewToHardware = false;
    [SerializeField]
    private bool logBaselinePreviewPackets = false;
    [SerializeField]
    [Tooltip("Baseline UDP 超过该时间没有收到可接受的新帧时，虚拟手回到全张开安全姿态。")]
    private float baselineUdpWatchdogTimeoutMs = 350f;
    [SerializeField]
    [Tooltip("拒绝来源时间戳过旧的本机 UDP 包，避免暂停/恢复后重放旧动作。")]
    private float baselineUdpMaxPacketAgeMs = 1000f;
    [SerializeField]
    private float baselineUdpMaxFutureSkewMs = 250f;
    [SerializeField]
    private int baselineUdpMaxPacketBytes = 32768;
    [SerializeField]
    [Tooltip("旧真机/机械臂入口的总开关。Phase 1/1.5 单右手 Unity 预览必须保持关闭。")]
    private bool allowLegacyHardwareControl = false;
    [SerializeField]
    private bool enableLegacyGestureSnapping = false;
    [SerializeField]
    private double lastBaselinePythonPipelineMs = -1.0;
    [SerializeField]
    private double lastBaselineUdpDeliveryMs = -1.0;
    [SerializeField]
    private double lastBaselineUnityQueueMs = -1.0;
    [SerializeField]
    private double lastBaselineSourceToTargetApplyMs = -1.0;
    [SerializeField]
    private int baselineOverwrittenPacketCount = 0;
    [SerializeField]
    private int baselineFrameGapCount = 0;
    [SerializeField]
    private int baselineRejectedPacketCount = 0;
    [SerializeField]
    private int baselineStalePacketCount = 0;
    [SerializeField]
    private int baselineWatchdogOpenCount = 0;

    private ConcurrentQueue<byte[]> incoming_messages = new ConcurrentQueue<byte[]>();
    private ConcurrentQueue<BaselinePreviewEnvelope> incomingBaselinePreviewPackets = new ConcurrentQueue<BaselinePreviewEnvelope>();
    // Start is called before the first frame update
    private RobotHand robotHand;
    private SVHFingerManager sVH;
    private RobotArm robotArm;
    private HandJointData dataReaded;
    private Vector3 headPosition, wristPosition;
    private Quaternion headRotation, wristRotation;
    private Vector3[] jointsPosition;
    private Quaternion[] jointsRotation;
    private double[] setRobotHandAngles;//设置的角度，从holo读取发送给灵巧手的 //分别是小拇指 无名指 中指 食指 大拇指弯曲  大拇指旋转
    private float[] setVirtualHandAngles;
    private float[] setVirtualSpread;
    private float[] getHandAngles;//获取的角度，从灵巧手获取发给虚拟手的。（暂时不读取）
    private bool isConnected, isSVH;
    private WebSocket ws;
    private string data;
    private byte[] ByteDate;
    private bool isDateUpdated;
    private bool a;
    private int b;
    private UdpClient baselineUdpClient;
    private Thread baselineUdpThread;
    private bool baselineUdpRunning;
    private readonly double[] baselinePreviewMaxRobotAngles = new double[] { 51.57, 56.72, 74.48, 42.97, 74.48, 45.84, 56.15, 56.15, 33.0 };
    private readonly float[] baselinePreviewMaxSpreadAngles = new float[] { 16f, 16f, 33f };
    private readonly float[] baselinePreviewChannelGains = new float[] { 1.00f, 1.00f, 1.00f, 1.00f, 1.12f, 1.12f, 1.18f, 1.20f, 1.00f };
    private int lastBaselineFrameIndex = -1;
    private double lastBaselineSourceTimestampMs = -1.0;
    private float lastBaselineAcceptedRealtime = -1f;
    private bool baselineSafeOpenApplied = false;
    private bool baselineHardwareWarningLogged = false;

    // Baseline UDP 是虚拟预览通道，不允许通过 Inspector 误把网络包转发到真机。
    private const bool BaselineUdpHardwareForwardingCompiled = false;

    private class BaselinePreviewEnvelope
    {
        public string json;
        public double receiveUnixMs;
    }

    [Serializable]
    private class BaselineSvhPreviewPacket
    {
        public bool enabled;
        public bool valid;
        public float[] target_positions;
    }

    [Serializable]
    private class BaselineFingerMapPacket
    {
        public float thumb;
        public float index;
        public float middle;
        public float ring;
        public float little;
    }

    [Serializable]
    private class BaselineControlRepresentationPacket
    {
        public bool valid;
        public bool features_valid;
        public bool command_ready;
        public float grasp_close;
        public float thumb_index_proximity;
        public float effective_pinch_strength;
        public float pinch_strength;
        public float support_flex;
        public BaselineFingerMapPacket finger_flex;
    }

    [Serializable]
    private class BaselineTimingPacket
    {
        public int schema_version;
        public string clock;
        public double source_read_start_unix_ms;
        public double source_read_end_unix_ms;
        public double detection_end_unix_ms;
        public double baseline_end_unix_ms;
        public double preview_end_unix_ms;
        public double payload_ready_unix_ms;
        public double udp_send_attempt_unix_ms;
    }

    [Serializable]
    private class BaselineFramePayloadPacket
    {
        public int frame_index;
        public double timestamp;
        public bool detected;
        public bool control_ready;
        public string gesture_stable;
        public float hand_open_ratio;
        public float pinch_distance_norm;
        public float latency_ms;
        public BaselineFingerMapPacket finger_curl;
        public BaselineControlRepresentationPacket control_representation;
        public BaselineSvhPreviewPacket svh_preview;
        public BaselineTimingPacket timing;
    }

    void Start()
    {
        jointsPosition = new Vector3[26];
        jointsRotation = new Quaternion[26];
        setRobotHandAngles = new double[(int)SVHConstants.SVHChannel.eSVH_DIMENSION];
        setRobotHandAngles = new double[(int)SVHConstants.SVHChannel.eSVH_DIMENSION];
        setVirtualHandAngles = new float[11];
        setVirtualSpread = new float[3];
        getHandAngles = new float[11];
        isConnected = false;
        ApplyBaselineSafeOpen("启动默认安全姿态", false);
        TryStartBaselineUdpPreview();
    }

    // Update is called once per frame
    void FixedUpdate()
    {


        if (isConnected && incoming_messages.TryDequeue(out var message))
        //if (incoming_messages.TryDequeue(out var message))
        {

            //var currentArmAngle = robotArm.GetAngle();
            var buf = new ByteBuffer(message);
            dataReaded = HandJointData.GetRootAsHandJointData(buf);
            var head = dataReaded.Head; //头部
            headPosition = new Vector3(head.Value.X, head.Value.Y, head.Value.Z);
            headRotation = new Quaternion(head.Value.Rx, head.Value.Ry, head.Value.Rz,head.Value.Rw);
            var wrist = dataReaded.Joints(1); //手腕作为原点
            wristPosition = new Vector3(wrist.Value.X, wrist.Value.Y, wrist.Value.Z);
            wristRotation = new Quaternion(wrist.Value.Rx, wrist.Value.Ry, wrist.Value.Rz,wrist.Value.Rw);

            for (int i = 0; i < 26; ++i)
            {
                var joint = dataReaded.Joints(i);
                jointsPosition[i] = new Vector3(joint.Value.X, joint.Value.Y, joint.Value.Z);
                jointsRotation[i] = new Quaternion(joint.Value.Rx, joint.Value.Ry, joint.Value.Rz,joint.Value.Rw);
            }
            CalRobotHandAngles();

            var setArmP = wristPosition - headPosition;
            //string s = setArmP.x.ToString() + "," + setArmP.y.ToString() + "," + setArmP.z.ToString();
            //string s = setRobotHandAngles[0].ToString() + "," + setRobotHandAngles[1].ToString()+ ","+setRobotHandAngles[2].ToString() +","+
            //            setRobotHandAngles[3].ToString() + "," + setRobotHandAngles[4].ToString() + ","+setRobotHandAngles[5].ToString() + "," +
            //            setRobotHandAngles[6].ToString() + "," + setRobotHandAngles[7].ToString() + ","+setRobotHandAngles[8].ToString() ;
            //Debug.Log(s);
            //setRobotHandAngles[0] = 0;
            //加入判断实现特定手势

            // 这组模板吸附来自旧的离散手势演示链，会把连续关节角强行压回少数固定姿态。
            // 现在默认关闭，避免干扰 Baseline -> Unity 的连续 UDP 预览。
            if (enableLegacyGestureSnapping)
            {
                if (setRobotHandAngles[0] > 30 && setRobotHandAngles[1] > 0 && setRobotHandAngles[2] > 50 &&
                    setRobotHandAngles[3] > 30 && setRobotHandAngles[4] > 50 && setRobotHandAngles[5] > 30 &&
                    setRobotHandAngles[6] > 35 && setRobotHandAngles[7] > 35 && setRobotHandAngles[8] > 0)
                {
                    Debug.Log("手势0");
                    setRobotHandAngles[0] = 20;
                    setRobotHandAngles[1] = 0;
                    setRobotHandAngles[2] = 74;
                    setRobotHandAngles[3] = 42;
                    setRobotHandAngles[4] = 74;
                    setRobotHandAngles[5] = 45;
                    setRobotHandAngles[6] = 56;
                    setRobotHandAngles[7] = 56;
                    setRobotHandAngles[8] = 10;
                }
                else if (setRobotHandAngles[0] > 30 && setRobotHandAngles[1] > 0 && setRobotHandAngles[2] > 50 &&
                    setRobotHandAngles[3] < 30 && setRobotHandAngles[4] > 50 && setRobotHandAngles[5] > 30 &&
                    setRobotHandAngles[6] > 35 && setRobotHandAngles[7] > 35 && setRobotHandAngles[8] > 0)
                {
                    Debug.Log("手势9");
                    setRobotHandAngles[0] = 20;
                    setRobotHandAngles[1] = 0;
                    setRobotHandAngles[2] = 50;
                    setRobotHandAngles[3] = 0;
                    setRobotHandAngles[4] = 74;
                    setRobotHandAngles[5] = 45;
                    setRobotHandAngles[6] = 56;
                    setRobotHandAngles[7] = 56;
                    setRobotHandAngles[8] = 10;
                }
                else if (setRobotHandAngles[0] > 30 && setRobotHandAngles[1] < 20 && setRobotHandAngles[2] < 30 &&
                    setRobotHandAngles[3] < 30 && setRobotHandAngles[4] < 30 && setRobotHandAngles[5] < 30 &&
                    setRobotHandAngles[6] > 35 && setRobotHandAngles[7] > 35 && setRobotHandAngles[8] > 0)
                {
                    Debug.Log("手势2");
                    setRobotHandAngles[0] = 20;
                    setRobotHandAngles[1] = 20;
                    setRobotHandAngles[2] = 0;
                    setRobotHandAngles[3] = 0;
                    setRobotHandAngles[4] = 0;
                    setRobotHandAngles[5] = 0;
                    setRobotHandAngles[6] = 56;
                    setRobotHandAngles[7] = 56;
                    setRobotHandAngles[8] = 0;
                }
                else if (setRobotHandAngles[0] > 0 && setRobotHandAngles[1] > 20 && setRobotHandAngles[2] < 35 &&
                    setRobotHandAngles[3] > 30 && setRobotHandAngles[4] < 35 && setRobotHandAngles[5] > 30 &&
                    setRobotHandAngles[6] > 35 && setRobotHandAngles[7] > 35 && setRobotHandAngles[8] > 0)
                {
                    Debug.Log("手势7");
                    setRobotHandAngles[0] = 25;
                    setRobotHandAngles[1] = 40;
                    setRobotHandAngles[2] = 20;
                    setRobotHandAngles[3] = 50;
                    setRobotHandAngles[4] = 20;
                    setRobotHandAngles[5] = 50;
                    setRobotHandAngles[6] = 56;
                    setRobotHandAngles[7] = 56;
                    setRobotHandAngles[8] = 0;
                }
            }


            ApplyRobotHandTargets(true);


            //robotArm
            if(ws !=null)
            {
                Target target = new Target(setArmP, wristRotation);
                string sendJson = JsonMapper.ToJson(target);
                ws.Send(sendJson);
            }

            
            /*            var angle = wristRotation.ToEulerAngles();
                        //float[] t = new float[] { setArmP.x, setArmP.y, setArmP.z, wristRotation.x, wristRotation.y, wristRotation.z, wristRotation.w };
                        float[] t = new float[] { -Mathf.Clamp(setArmP.z, 0.1f, 0.5f), Mathf.Clamp(setArmP.x, -0.3f, 0.3f), Mathf.Clamp(setArmP.y, -0.4f, 0.1f)+0.75f, angle.x,angle.y,angle.z };
                        string s = JsonMapper.ToJson(t);
                        ws.Send(s);*/
        }
        DrainBaselinePreviewPackets();
        CheckBaselineUdpWatchdog();
        while (incoming_messages.TryDequeue(out var result))
        { }
    }

    private void TryStartBaselineUdpPreview()
    {
        if (!enableBaselineUdpPreview || baselineUdpThread != null)
        {
            return;
        }

        try
        {
            // 只绑定 IPv4 loopback；局域网其他主机不能向这个预览入口注入动作。
            baselineUdpClient = new UdpClient(AddressFamily.InterNetwork);
            baselineUdpClient.Client.ExclusiveAddressUse = true;
            baselineUdpClient.Client.Bind(new IPEndPoint(IPAddress.Loopback, baselineUdpListenPort));
            baselineUdpClient.Client.ReceiveTimeout = 250;
            baselineUdpRunning = true;
            baselineUdpThread = new Thread(BaselineUdpListenLoop);
            baselineUdpThread.IsBackground = true;
            baselineUdpThread.Start();
            Debug.Log("Baseline UDP 预览监听已启动，仅绑定 127.0.0.1:" + baselineUdpListenPort);
            if (applyBaselinePreviewToHardware)
            {
                Debug.LogWarning("applyBaselinePreviewToHardware 已被 Phase 1.5 安全门永久忽略；UDP 只驱动虚拟手。");
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("启动 Baseline UDP 预览监听失败：" + ex.Message);
            baselineUdpRunning = false;
            baselineUdpClient = null;
            baselineUdpThread = null;
        }
    }

    private void BaselineUdpListenLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (baselineUdpRunning)
        {
            try
            {
                byte[] packet = baselineUdpClient.Receive(ref remote);
                if (packet == null || packet.Length == 0 || packet.Length > Math.Max(1024, baselineUdpMaxPacketBytes))
                {
                    Interlocked.Increment(ref baselineRejectedPacketCount);
                    continue;
                }
                string json = Encoding.UTF8.GetString(packet);
                while (incomingBaselinePreviewPackets.TryDequeue(out _))
                {
                    Interlocked.Increment(ref baselineOverwrittenPacketCount);
                }
                incomingBaselinePreviewPackets.Enqueue(new BaselinePreviewEnvelope
                {
                    json = json,
                    receiveUnixMs = UtcNowUnixMs()
                });
                if (logBaselinePreviewPackets)
                {
                    Debug.Log("收到 Baseline UDP 预览数据，长度=" + packet.Length + "，来源=" + remote.ToString());
                }
            }
            catch (SocketException ex)
            {
                if (ex.SocketErrorCode == SocketError.TimedOut || ex.SocketErrorCode == SocketError.Interrupted)
                {
                    continue;
                }
                if (baselineUdpRunning)
                {
                    Debug.LogWarning("接收 Baseline UDP 预览数据失败：" + ex.Message);
                }
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception ex)
            {
                if (baselineUdpRunning)
                {
                    Debug.LogWarning("接收 Baseline UDP 预览数据失败：" + ex.Message);
                }
            }
        }
    }

    private void DrainBaselinePreviewPackets()
    {
        BaselinePreviewEnvelope latestPacket = null;
        while (incomingBaselinePreviewPackets.TryDequeue(out var packet))
        {
            latestPacket = packet;
        }

        if (latestPacket == null || string.IsNullOrEmpty(latestPacket.json))
        {
            return;
        }

        TryApplyBaselinePreviewPacket(latestPacket.json, latestPacket.receiveUnixMs);
    }

    private void CheckBaselineUdpWatchdog()
    {
        if (!enableBaselineUdpPreview || baselineSafeOpenApplied || lastBaselineAcceptedRealtime < 0f)
        {
            return;
        }

        float timeoutMs = Mathf.Max(50f, baselineUdpWatchdogTimeoutMs);
        if ((Time.realtimeSinceStartup - lastBaselineAcceptedRealtime) * 1000f >= timeoutMs)
        {
            baselineWatchdogOpenCount += 1;
            ApplyBaselineSafeOpen("UDP 失联 watchdog 超时", true);
        }
    }

    private void TryApplyBaselinePreviewPacket(string json, double receiveUnixMs)
    {
        try
        {
            BaselineFramePayloadPacket packet = JsonUtility.FromJson<BaselineFramePayloadPacket>(json);
            if (packet == null)
            {
                baselineRejectedPacketCount += 1;
                return;
            }

            if (!TryAcceptBaselinePacketOrder(packet, receiveUnixMs))
            {
                return;
            }

            lastBaselineAcceptedRealtime = Time.realtimeSinceStartup;

            if (packet.svh_preview != null)
            {
                BaselineControlRepresentationPacket control = packet.control_representation;
                bool canonicalValid =
                    packet.detected &&
                    packet.control_ready &&
                    packet.svh_preview.enabled &&
                    packet.svh_preview.valid &&
                    control != null &&
                    control.valid &&
                    control.features_valid &&
                    control.command_ready;
                if (canonicalValid)
                {
                    float[] expandedTargets = ExpandPreviewTargets(packet.svh_preview.target_positions);
                    if (expandedTargets != null)
                    {
                        ApplyBaselinePreviewTargets(expandedTargets, BaselineUdpHardwareForwardingCompiled);
                        baselineSafeOpenApplied = false;
                        RecordBaselineTimingDiagnostics(packet, receiveUnixMs);

                        if (logBaselinePreviewPackets)
                        {
                            Debug.Log("已应用 Baseline SVH preview 目标：" + string.Join(", ", expandedTargets));
                        }
                        return;
                    }
                }

                // canonical payload 已经显式包含 preview 时，invalid 就是安全门信号；
                // 不允许再落入旧连续特征兜底，从而绕过 valid/control_ready。
                ApplyBaselineSafeOpen("收到 invalid/no-hand Baseline 帧", true);
                RecordBaselineTimingDiagnostics(packet, receiveUnixMs);
                return;
            }

            // 仅兼容真正缺少 svh_preview 字段的旧 payload；同样严格检查所有控制门。
            if (TryApplyContinuousBaselineFrame(packet, BaselineUdpHardwareForwardingCompiled))
            {
                baselineSafeOpenApplied = false;
                RecordBaselineTimingDiagnostics(packet, receiveUnixMs);
                if (logBaselinePreviewPackets)
                {
                    Debug.Log("已应用 Baseline 连续特征兜底目标。");
                }
                return;
            }


            ApplyBaselineSafeOpen("旧 payload 未通过控制有效性检查", true);
            RecordBaselineTimingDiagnostics(packet, receiveUnixMs);
        }
        catch (Exception ex)
        {
            baselineRejectedPacketCount += 1;
            Debug.LogWarning("解析 Baseline UDP 预览数据失败：" + ex.Message);
        }
    }

    private bool TryAcceptBaselinePacketOrder(BaselineFramePayloadPacket packet, double receiveUnixMs)
    {
        if (packet.frame_index < 0 || double.IsNaN(packet.timestamp) || double.IsInfinity(packet.timestamp) || packet.timestamp <= 0.0)
        {
            baselineRejectedPacketCount += 1;
            return false;
        }

        double sourceTimestampMs = packet.timestamp * 1000.0;
        double packetAgeMs = receiveUnixMs - sourceTimestampMs;
        if (packetAgeMs > Math.Max(100.0, baselineUdpMaxPacketAgeMs) ||
            packetAgeMs < -Math.Max(0.0, baselineUdpMaxFutureSkewMs))
        {
            baselineStalePacketCount += 1;
            return false;
        }

        bool pythonRestart =
            packet.frame_index == 0 &&
            lastBaselineFrameIndex > 0 &&
            sourceTimestampMs > lastBaselineSourceTimestampMs;
        if (lastBaselineFrameIndex >= 0 && !pythonRestart)
        {
            if (packet.frame_index <= lastBaselineFrameIndex || sourceTimestampMs <= lastBaselineSourceTimestampMs)
            {
                baselineStalePacketCount += 1;
                return false;
            }
            if (packet.frame_index > lastBaselineFrameIndex + 1)
            {
                baselineFrameGapCount += packet.frame_index - lastBaselineFrameIndex - 1;
            }
        }
        else if (pythonRestart)
        {
            baselineFrameGapCount = 0;
        }

        lastBaselineFrameIndex = packet.frame_index;
        lastBaselineSourceTimestampMs = sourceTimestampMs;
        return true;
    }

    private static double UtcNowUnixMs()
    {
        return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
    }

    private void RecordBaselineTimingDiagnostics(BaselineFramePayloadPacket packet, double receiveUnixMs)
    {
        double applyUnixMs = UtcNowUnixMs();
        lastBaselineUnityQueueMs = applyUnixMs - receiveUnixMs;

        if (packet.timing != null &&
            packet.timing.schema_version == 1 &&
            packet.timing.clock == "unix_epoch_ms" &&
            packet.timing.source_read_end_unix_ms > 0.0 &&
            packet.timing.udp_send_attempt_unix_ms > 0.0)
        {
            lastBaselinePythonPipelineMs =
                packet.timing.udp_send_attempt_unix_ms - packet.timing.source_read_end_unix_ms;
            lastBaselineUdpDeliveryMs =
                receiveUnixMs - packet.timing.udp_send_attempt_unix_ms;
            lastBaselineSourceToTargetApplyMs =
                applyUnixMs - packet.timing.source_read_end_unix_ms;
        }
        else
        {
            // 旧 payload 仍可使用，只是没有跨 Python/Unity 的阶段诊断。
            lastBaselinePythonPipelineMs = -1.0;
            lastBaselineUdpDeliveryMs = -1.0;
            lastBaselineSourceToTargetApplyMs = -1.0;
        }

        if (logBaselinePreviewPackets)
        {
            Debug.Log(
                "Baseline timing：Python=" + lastBaselinePythonPipelineMs.ToString("F3") +
                " ms，UDP=" + lastBaselineUdpDeliveryMs.ToString("F3") +
                " ms，Unity queue=" + lastBaselineUnityQueueMs.ToString("F3") +
                " ms，source->target apply=" + lastBaselineSourceToTargetApplyMs.ToString("F3") +
                " ms，覆盖包=" + baselineOverwrittenPacketCount +
                "，frame gap=" + baselineFrameGapCount +
                "，拒绝包=" + baselineRejectedPacketCount +
                "，过期/乱序包=" + baselineStalePacketCount +
                "，watchdog 张开=" + baselineWatchdogOpenCount
            );
        }
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static float NormalizeRange(float value, float openRef, float closedRef)
    {
        float denom = closedRef - openRef;
        if (Mathf.Abs(denom) < 1e-6f)
        {
            return 0f;
        }
        return Mathf.Clamp01((value - openRef) / denom);
    }

    private static float SafeFingerFlex(BaselineFingerMapPacket map, Func<BaselineFingerMapPacket, float> selector)
    {
        if (map == null)
        {
            return 0f;
        }
        float value = selector(map);
        return IsFinite(value) ? Mathf.Clamp01(value) : 0f;
    }

    private bool TryApplyContinuousBaselineFrame(BaselineFramePayloadPacket packet, bool driveHardware)
    {
        if (packet == null || !packet.detected || !packet.control_ready)
        {
            return false;
        }

        if (packet.control_representation == null ||
            !packet.control_representation.valid ||
            !packet.control_representation.features_valid ||
            !packet.control_representation.command_ready ||
            packet.control_representation.finger_flex == null)
        {
            return false;
        }

        BaselineControlRepresentationPacket control = packet.control_representation;
        BaselineFingerMapPacket fingerFlex = control.finger_flex;

        float thumbFlex = SafeFingerFlex(fingerFlex, value => value.thumb);
        float indexFlex = SafeFingerFlex(fingerFlex, value => value.index);
        float middleFlex = SafeFingerFlex(fingerFlex, value => value.middle);
        float ringFlex = SafeFingerFlex(fingerFlex, value => value.ring);
        float littleFlex = SafeFingerFlex(fingerFlex, value => value.little);

        float graspClose = Mathf.Clamp01(control.grasp_close);
        float pinchClose = Mathf.Clamp01(Mathf.Max(control.thumb_index_proximity, control.effective_pinch_strength));
        float supportFlex = Mathf.Clamp01(control.support_flex);
        if (!IsFinite(control.grasp_close) ||
            !IsFinite(control.thumb_index_proximity) ||
            !IsFinite(control.effective_pinch_strength) ||
            !IsFinite(control.support_flex) ||
            !IsFinite(packet.hand_open_ratio))
        {
            return false;
        }
        float openAmount = NormalizeRange(packet.hand_open_ratio, 0.25f, 0.95f);
        float spread = Mathf.Clamp01((0.75f * openAmount + 0.25f * (1f - graspClose)) * (1f - 0.25f * pinchClose));

        float[] normalizedTargets = new float[]
        {
            thumbFlex,
            Mathf.Clamp01(Mathf.Max(0.45f * thumbFlex, 0.95f * pinchClose)),
            Mathf.Clamp01(0.70f * indexFlex + 0.30f * pinchClose),
            indexFlex,
            Mathf.Clamp01(Mathf.Max(middleFlex, 0.35f * supportFlex)),
            middleFlex,
            ringFlex,
            littleFlex,
            spread,
        };

        ApplyBaselinePreviewTargets(normalizedTargets, driveHardware);
        return true;
    }

    private float[] ExpandPreviewTargets(float[] source)
    {
        if (source == null)
        {
            return null;
        }

        if (source.Length == 9)
        {
            float[] expanded = new float[9];
            for (int i = 0; i < 9; ++i)
            {
                if (!IsFinite(source[i]))
                {
                    return null;
                }
                expanded[i] = Mathf.Clamp01(source[i]);
            }
            return expanded;
        }

        if (source.Length == 5)
        {
            for (int i = 0; i < 5; ++i)
            {
                if (!IsFinite(source[i]))
                {
                    return null;
                }
            }
            float thumb = Mathf.Clamp01(source[0]);
            float index = Mathf.Clamp01(source[1]);
            float middle = Mathf.Clamp01(source[2]);
            float ring = Mathf.Clamp01(source[3]);
            float pinky = Mathf.Clamp01(source[4]);
            return new float[]
            {
                thumb,
                thumb,
                index,
                index,
                middle,
                middle,
                ring,
                pinky,
                0f
            };
        }

        return null;
    }

    public void ApplyBaselinePreviewTargets(float[] normalizedTargets, bool driveHardware = false)
    {
        if (normalizedTargets == null || normalizedTargets.Length < 9)
        {
            return;
        }

        for (int i = 0; i < 9; ++i)
        {
            if (!IsFinite(normalizedTargets[i]))
            {
                return;
            }
            float normalized = Mathf.Clamp01(normalizedTargets[i] * baselinePreviewChannelGains[i]);
            setRobotHandAngles[i] = normalized * baselinePreviewMaxRobotAngles[i];
        }

        float spread = Mathf.Clamp01(normalizedTargets[8]);
        setVirtualSpread[0] = spread * baselinePreviewMaxSpreadAngles[0];
        setVirtualSpread[1] = spread * baselinePreviewMaxSpreadAngles[1];
        setVirtualSpread[2] = spread * baselinePreviewMaxSpreadAngles[2];

        ApplyRobotHandTargets(driveHardware);
    }

    private void ApplyRobotHandTargets(bool driveHardware)
    {
        if (driveHardware && !allowLegacyHardwareControl)
        {
            if (!baselineHardwareWarningLogged)
            {
                Debug.LogWarning("旧真机控制总开关未启用；本次目标只应用到 Unity 虚拟手。");
                baselineHardwareWarningLogged = true;
            }
            driveHardware = false;
        }
        if (driveHardware && sVH != null)
        {
            sVH.setAllTargetPositions(setRobotHandAngles);
        }

        if (virtualSVH == null || virtualSVH.Length < 20)
        {
            return;
        }

        float[] tempAngle = new float[]
        {
            (float)(setRobotHandAngles[0]) / 51.57f, (float)(setRobotHandAngles[1]) / 56.72f, (float)(setRobotHandAngles[2]) / 74.48f,
            (float)(setRobotHandAngles[3]) / 42.97f, (float)(setRobotHandAngles[4]) / 74.48f, (float)(setRobotHandAngles[5]) / 45.84f,
            (float)(setRobotHandAngles[6]) / 56.15f, (float)(setRobotHandAngles[7]) / 56.15f, 0f
        };
        float[] virAng = new float[]
        {
            tempAngle[0] * 55.59983f, tempAngle[0] * 56.43978f, tempAngle[0] * 80.55787f,
            tempAngle[1] * 56.6025f, tempAngle[1] * 56.60021f,
            tempAngle[2] * 76.43257f, tempAngle[2] * 79.87032f,
            tempAngle[3] * 45.75011f,
            tempAngle[4] * 76.43257f, tempAngle[4] * 79.87032f,
            tempAngle[5] * 45.75011f,
            tempAngle[6] * 56.25013f, tempAngle[6] * 76.43257f, tempAngle[6] * 79.92761f,
            tempAngle[7] * 56.25013f, tempAngle[7] * 76.43257f, tempAngle[7] * 79.92761f,
            setVirtualSpread[0], setVirtualSpread[1], setVirtualSpread[2]
        };
        ArticulationDrive[] ad = new ArticulationDrive[20];
        for (int i = 0; i < 20; ++i)
        {
            ad[i] = virtualSVH[i].xDrive;
            ad[i].target = virAng[i];
            virtualSVH[i].xDrive = ad[i];
        }
    }

    private void ApplyBaselineSafeOpen(string reason, bool logTransition)
    {
        if (baselineSafeOpenApplied)
        {
            return;
        }
        ApplyBaselinePreviewTargets(new float[9], BaselineUdpHardwareForwardingCompiled);
        baselineSafeOpenApplied = true;
        if (logTransition)
        {
            Debug.LogWarning("Baseline UDP 已切换到虚拟手安全张开姿态：" + reason);
        }
    }

    private void CalRobotHandAngles()//没有传0-None，和MRTK的编号会有区别
    {
        double[] temp = new double[(int)SVHConstants.SVHChannel.eSVH_DIMENSION];

        temp[0] = CalFingerAngle(2,3,3,4,0,51);//大拇指
        temp[1] = CalOppostitionAngle();//CalFingerAngle();//大拇指旋转（整个手捏紧）
        temp[2] = CalFingerAngle(7,8,8,9,0,74);//食指远端
        temp[3] = CalFingerAngle(6,7,7,8,0,42);//食指近端
        temp[4] = CalFingerAngle(13,14,14,15,0,74);//中指远端
        temp[5] = CalFingerAngle(11,12,12,13,0,42);//中指近端
        temp[6] = CalFingerAngle(16,17,17,18,0,56);//无名指
        temp[7] = CalFingerAngle(21,22,22,23,0,56);//小拇指
        temp[8] = CalSpreadAngle();// CalFingerAngle();//手指分开

        for(int i =0;i<(int)SVHConstants.SVHChannel.eSVH_DIMENSION;++i)
        {
            setRobotHandAngles[i] = Math.Abs(temp[i] - setRobotHandAngles[i]) > 1.0 ? temp[i] : setRobotHandAngles[i];
        }

    }

    private double CalFingerAngle(int joint1Coor, int joint2Coor, int joint3Coor, int joint4Coor, int min, int max)
    {
        var J2J1 = jointsPosition[joint1Coor] - jointsPosition[joint2Coor];
        var J3J2 = jointsPosition[joint3Coor] - jointsPosition[joint4Coor];
        var angle = Vector3.Angle(J2J1, J3J2);
        return (Mathf.Clamp(angle, min, max));
    }
    
    private double CalSpreadAngle()
    {
        setVirtualSpread[0] = (float)CalFingerAngle(7,  8,  12, 13, 0, 16);
        setVirtualSpread[1] = (float)CalFingerAngle(17, 18, 13, 14, 0, 16);
        setVirtualSpread[2] = (float)CalFingerAngle(22, 23, 14, 15, 0, 33);
        return (setVirtualSpread[0] + setVirtualSpread[1] + setVirtualSpread[2] / 2f) / 3f * 2;
    }
    
    private double CalOppostitionAngle()
    {
        var line1 = jointsPosition[0] - jointsPosition[7];
        var line2 = jointsPosition[0] - jointsPosition[22];
        var line3 = jointsPosition[2] - jointsPosition[3];
        var normal = Vector3.Cross(line1, line2);
        var angle = Vector3.Angle(line3, normal);
        return Mathf.Clamp(90 - angle, 0, 45);

    }

    private void CalVirtualHandAngles()
    {

        for (int i = 0; i < 4; ++i) //4个拇指 i=0-4 ->食指、中指、无名指、小拇指
        {
            //外
            setVirtualHandAngles[6 - i * 2] = -Vector3.Angle(jointsPosition[7 + 5 * i] - jointsPosition[8 + 5 * i], jointsPosition[8 + 5 * i] - jointsPosition[10 + 5 * i]);
            //内
            setVirtualHandAngles[7 - i * 2] = -Vector3.Angle(jointsPosition[6 + 5 * i] - jointsPosition[7 + 5 * i], jointsPosition[7 + 5 * i] - jointsPosition[8 + 5 * i]);
            
        }

        //大拇指外
        setVirtualHandAngles[8] = -Vector3.Angle(jointsPosition[3] - jointsPosition[4], jointsPosition[4] - jointsPosition[5]);
        //大拇指内
        setVirtualHandAngles[9] = Vector3.Angle(jointsPosition[2] - jointsPosition[3], jointsPosition[3] - jointsPosition[4]);
        //大拇指旋转
        setVirtualHandAngles[10] = Vector3.Angle(jointsPosition[0] - jointsPosition[5], jointsPosition[0] - jointsPosition[7]);
    }

    public void Init(string COM,string IP,bool isSVHorInspire)
    {
        isConnected = false;
        isSVH = isSVHorInspire;
        if (!allowLegacyHardwareControl && (!string.IsNullOrEmpty(COM) || !string.IsNullOrEmpty(IP)))
        {
            Debug.LogWarning("Phase 1/1.5 默认隔离旧真机、串口和机械臂网络入口；已忽略本次 COM/IP 参数。");
            COM = string.Empty;
            IP = string.Empty;
        }
        if (COM.Length == 0)
        {
            robotHand = null;
        }
        else
        {
            if(isSVH)
            {
                sVH = new SVHFingerManager(COM,2000);
            }
            else
            {
                robotHand = new RobotHand(COM, 115200);
            }

        }

        /*        if (IP.Length == 0)
                {
                    robotArm = null;
                }
                else
                {
                    robotArm = new RobotArm(IP, 8899);
                }*/
        if (IP.Length == 0)
        {
            ws = null;
        }
        else
        {
            ConnectServer(IP);
        }

        if (robotHand != null) { robotHand.SetAngle(new short[] { 1000, 1000, 1000, 1000, 1000, 1000 }); }
        while (incoming_messages.TryDequeue( out var result))
        { }
        isConnected = true;

    }
    
    void ConnectServer(string url)
    {
        url = "ws://" +url+ ":8080";
        //Debug.Log(url);
        ws = new WebSocket(url);
        ws.OnMessage += (sender, e) =>
        {
            if (e.IsText)
            {
                data = e.Data;
            }
            else if (e.IsBinary)
            {
                ByteDate = e.RawData;
                isDateUpdated = true;
            }
        };
        ws.OnOpen += (sender, e) =>
        {
            Debug.Log("Connected");
            isConnected = true;
        };
        ws.OnError += (sender, e) =>
        {
            Debug.Log(e.ToString());
        };

        ws.Connect();
    }

    public void OnDisconnected()
    {
        isConnected = false;
        if (robotHand != null)
        {
            robotHand.Stop();
        }
        if (robotArm != null)
        {
            robotArm.Stop();
        }
        robotHand = null;
        robotArm = null;
    }

    public void OnData(byte[] data)
    {
        incoming_messages.Enqueue(data);
    }

    public void OnClicked()
    {
        setRobotHandAngles = new double[] { 25, 40, 20, 50, 20,50, 56, 56, 0 };
        ApplyRobotHandTargets(true);
    }

    public void OnDestroy()
    {
        isConnected = false;
        baselineUdpRunning = false;
        if (baselineUdpClient != null)
        {
            baselineUdpClient.Close();
            baselineUdpClient = null;
        }
        if (baselineUdpThread != null && baselineUdpThread.IsAlive)
        {
            baselineUdpThread.Join(200);
        }
        if (robotHand != null)
        {
            robotHand.Stop();
        }
        if (robotArm != null)
        {
            robotArm.Stop();
        }
        if(sVH != null)
        {
            sVH.Stop();
        }
        if (ws != null)
        {
            ws.CloseAsync();
        }

    }

}

public class Target
{
    public Vec3 pos;
    public Vec4 ori;

    public class Vec3
    {
        public float x;
        public float y;
        public float z;
    }
    
    public class Vec4
    {
        public float x;
        public float y;
        public float z;
        public float w;
    }

    public Target(Vector3 position, Quaternion rotation)
    {
        pos = new Vec3();
        ori = new Vec4();
        pos.x = position.x;
        pos.y = position.y;
        pos.z = position.z;
        
        ori.x = rotation.x;
        ori.y = rotation.y;
        ori.z = rotation.z;
        ori.w = rotation.w;
    }

}
