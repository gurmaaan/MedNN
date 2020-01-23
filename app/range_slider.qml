import QtQuick.Controls 2.10

RangeSlider {
    from: 0
    to: 100
    first.value: 60
    second.value: 80
    first.onMoved: {
        mainWindow.update_test_size(first.value, second.value)
    }
    second.onMoved: {
        mainWindow.update_test_size(first.value, second.value)
    }
}
