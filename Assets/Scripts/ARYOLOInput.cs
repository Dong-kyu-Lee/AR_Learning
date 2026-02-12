using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class ARYOLOInput : MonoBehaviour
{
    [Header("AR Components")]
    public ARCameraManager cameraManager;
    
    [Header("YOLO Reference")]
    public RunYOLO yoloProcessor; // 기존 RunYOLO 스크립트 참조

    private RenderTexture arTexture;

    void OnEnable()
    {
        // 카메라 프레임이 업데이트될 때마다 호출될 이벤트 연결
        cameraManager.frameReceived += OnCameraFrameReceived;
    }

    void OnDisable()
    {
        cameraManager.frameReceived -= OnCameraFrameReceived;
    }

    void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        // 1. AR 카메라의 최신 텍스처(GPU)를 가져옵니다.
        // AR Foundation은 여러 개의 텍스처(Y, CbCr 등)를 사용할 수 있으므로 
        // 하드웨어 가속이 적용된 '_MainTex' 혹은 'propertyNameIds'를 활용합니다.
        
        if (!cameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
        {
            return;
        }

        // 실무적 타협: CPU 이미지를 매번 변환하는 것보다 
        // 화면에 그려지고 있는 AR Background의 텍스처를 직접 사용하는 것이 성능에 유리합니다.
        // 여기서는 가장 보편적이고 빠른 'Texture2D' 기반 방식을 제안합니다.
    }

    // Update문에서 RunYOLO에 텍스처를 공급하는 방식
    void Update()
    {
        // ARCameraBackground에서 현재 화면에 렌더링 중인 텍스처를 가져오는 로직
        // 보통 AR Foundation은 내부 머티리얼을 통해 화면을 그리므로, 
        // 아래와 같이 메인 카메라의 스카이박스나 전용 API를 통해 접근합니다.
    }
}
