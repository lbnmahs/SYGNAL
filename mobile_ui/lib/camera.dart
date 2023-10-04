import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:mobile_ui/controller/controller.dart';

class CameraWidget extends StatelessWidget {
  const CameraWidget({ super.key });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GetBuilder<Controller>(
        init: Controller(),
        builder: (controller) {
          return controller.isCameraInitialized.value 
            ?
            AspectRatio(
              aspectRatio: controller.cameraController.value.aspectRatio,
              child: CameraPreview(controller.cameraController),
            ):
            const Center(
              child: CircularProgressIndicator(),
            );

        }
      ),
    );
  }
}