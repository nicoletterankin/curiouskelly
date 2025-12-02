sub init()
    m.webView = m.top.findNode("webView")
    m.loadingSpinner = m.top.findNode("loadingSpinner")
    
    ' Set up WebView observers
    m.webView.observeField("state", "onWebViewStateChange")
    m.webView.observeField("loadStatus", "onLoadStatusChange")
    
    ' Set focus to WebView
    m.webView.setFocus(true)
end sub

sub onWebViewStateChange()
    state = m.webView.state
    print "WebView state: "; state
end sub

sub onLoadStatusChange()
    loadStatus = m.webView.loadStatus
    print "Load status: "; loadStatus
    
    if loadStatus = "ready"
        ' Hide loading spinner when page loads
        m.loadingSpinner.visible = false
    else if loadStatus = "loading"
        ' Show loading spinner
        m.loadingSpinner.visible = true
    else if loadStatus = "failed"
        ' Handle load failure
        print "Failed to load page"
        m.loadingSpinner.visible = false
    end if
end sub





