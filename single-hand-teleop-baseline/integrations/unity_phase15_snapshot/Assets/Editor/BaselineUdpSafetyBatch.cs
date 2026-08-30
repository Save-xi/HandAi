using System;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Phase 1.5 的无界面行为回归。只构造 RobotControlScript，不连接真机。
/// 失败时抛异常，让 Unity batchmode 返回非零；成功时输出固定 PASS 标记。
/// </summary>
public static class BaselineUdpSafetyBatch
{
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;

    private static FieldInfo Field(string name)
    {
        FieldInfo field = typeof(RobotControlScript).GetField(name, PrivateInstance);
        if (field == null)
        {
            throw new MissingFieldException(typeof(RobotControlScript).FullName, name);
        }
        return field;
    }

    private static MethodInfo Method(string name)
    {
        MethodInfo method = typeof(RobotControlScript).GetMethod(name, PrivateInstance);
        if (method == null)
        {
            throw new MissingMethodException(typeof(RobotControlScript).FullName, name);
        }
        return method;
    }

    private static void Set(RobotControlScript target, string name, object value)
    {
        Field(name).SetValue(target, value);
    }

    private static T Get<T>(RobotControlScript target, string name)
    {
        return (T)Field(name).GetValue(target);
    }

    private static object Invoke(RobotControlScript target, string name, params object[] args)
    {
        try
        {
            return Method(name).Invoke(target, args);
        }
        catch (TargetInvocationException ex) when (ex.InnerException != null)
        {
            throw ex.InnerException;
        }
    }

    private static string Packet(int frameIndex, double timestamp, bool valid, float target)
    {
        string ts = timestamp.ToString("R", CultureInfo.InvariantCulture);
        string positions = valid
            ? string.Join(",", Enumerable.Repeat(target.ToString("R", CultureInfo.InvariantCulture), 9))
            : string.Empty;
        return "{" +
            "\"frame_index\":" + frameIndex + "," +
            "\"timestamp\":" + ts + "," +
            "\"detected\":" + (valid ? "true" : "false") + "," +
            "\"control_ready\":" + (valid ? "true" : "false") + "," +
            "\"control_representation\":{" +
                "\"valid\":" + (valid ? "true" : "false") + "," +
                "\"features_valid\":" + (valid ? "true" : "false") + "," +
                "\"command_ready\":" + (valid ? "true" : "false") +
            "}," +
            "\"svh_preview\":{" +
                "\"enabled\":true," +
                "\"valid\":" + (valid ? "true" : "false") + "," +
                "\"target_positions\":[" + positions + "]" +
            "}" +
        "}";
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static bool AllZero(double[] values)
    {
        return values != null && values.Length >= 9 && values.Take(9).All(value => Math.Abs(value) < 1e-9);
    }

    public static void Run()
    {
        GameObject gameObject = new GameObject("Phase15BaselineUdpSafetyBatch");
        RobotControlScript component = gameObject.AddComponent<RobotControlScript>();
        try
        {
            Set(component, "enableBaselineUdpPreview", false);
            Invoke(component, "Start");
            Require(AllZero(Get<double[]>(component, "setRobotHandAngles")), "启动时未进入全张开安全姿态");

            double nowSeconds = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
            Invoke(component, "TryApplyBaselinePreviewPacket", Packet(0, nowSeconds, true, 0.5f), nowSeconds * 1000.0);
            double[] validTargets = Get<double[]>(component, "setRobotHandAngles");
            Require(validTargets.Take(8).All(value => value > 0.0), "有效 preview 未驱动虚拟目标");

            Invoke(component, "TryApplyBaselinePreviewPacket", Packet(1, nowSeconds + 0.001, false, 0f), (nowSeconds + 0.001) * 1000.0);
            Require(AllZero(Get<double[]>(component, "setRobotHandAngles")), "invalid/no-hand 帧没有立即安全张开");

            Invoke(component, "TryApplyBaselinePreviewPacket", Packet(2, nowSeconds + 0.002, true, 0.6f), (nowSeconds + 0.002) * 1000.0);
            double retained = Get<double[]>(component, "setRobotHandAngles")[0];
            Invoke(component, "TryApplyBaselinePreviewPacket", Packet(1, nowSeconds + 0.003, true, 0.1f), (nowSeconds + 0.003) * 1000.0);
            Require(Math.Abs(Get<double[]>(component, "setRobotHandAngles")[0] - retained) < 1e-9, "乱序包改写了目标");

            Invoke(component, "TryApplyBaselinePreviewPacket", Packet(3, nowSeconds - 10.0, true, 0.1f), nowSeconds * 1000.0);
            Require(Math.Abs(Get<double[]>(component, "setRobotHandAngles")[0] - retained) < 1e-9, "过期包改写了目标");
            Require(Get<int>(component, "baselineStalePacketCount") >= 2, "乱序/过期计数未更新");

            Set(component, "enableBaselineUdpPreview", true);
            Set(component, "baselineUdpWatchdogTimeoutMs", 50f);
            Set(component, "lastBaselineAcceptedRealtime", Time.realtimeSinceStartup - 1f);
            Invoke(component, "CheckBaselineUdpWatchdog");
            Require(AllZero(Get<double[]>(component, "setRobotHandAngles")), "watchdog 未回到安全张开");

            Set(component, "baselineUdpListenPort", 0);
            Invoke(component, "TryStartBaselineUdpPreview");
            UdpClient client = Get<UdpClient>(component, "baselineUdpClient");
            IPEndPoint endpoint = client.Client.LocalEndPoint as IPEndPoint;
            Require(endpoint != null && IPAddress.Loopback.Equals(endpoint.Address), "UDP socket 未绑定 IPv4 loopback");

            using (UdpClient sender = new UdpClient(AddressFamily.InterNetwork))
            {
                double udpTimestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
                byte[] validUdp = Encoding.UTF8.GetBytes(Packet(3, udpTimestamp, true, 0.4f));
                sender.Send(validUdp, validUdp.Length, endpoint);
                Thread.Sleep(80);
                Invoke(component, "DrainBaselinePreviewPackets");
                Require(Get<double[]>(component, "setRobotHandAngles")[0] > 0.0, "loopback UDP 有效包未应用");

                byte[] invalidUdp = Encoding.UTF8.GetBytes(Packet(4, udpTimestamp + 0.001, false, 0f));
                sender.Send(invalidUdp, invalidUdp.Length, endpoint);
                Thread.Sleep(80);
                Invoke(component, "DrainBaselinePreviewPackets");
                Require(AllZero(Get<double[]>(component, "setRobotHandAngles")), "loopback UDP invalid 包未安全张开");
            }

            Debug.Log("PHASE15_UNITY_SAFETY_BATCH_PASS");
        }
        finally
        {
            component.OnDestroy();
            UnityEngine.Object.DestroyImmediate(gameObject);
        }
    }
}
