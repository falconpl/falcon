//
//  gtk_ScrolledWindow.h


#ifndef GTK_SCROLLED_WINDOW_HPP

#define GTK_SCROLLED_WINDOW_HPP

#include "modgtk.hpp"


namespace Falcon {

namespace Gtk {

/**
 * \class Falcon::Gtk::ScrolledWindow
 */

class ScrolledWindow : public Gtk::CoreGObject {
    
public:
    
    ScrolledWindow( const Falcon::CoreClass *, const GtkScrolledWindow * = 0);
    
    static Falcon::CoreObject *factory( const Falcon::CoreClass *, void *, bool );
    
    static void modInit( Falcon::Module * );
    
    static FALCON_FUNC init( VMARG );
    
    static FALCON_FUNC set_policy( VMARG );
    
    static FALCON_FUNC get_policy( VMARG );
    
    static FALCON_FUNC add_with_viewport( VMARG );
    
    static FALCON_FUNC get_placement( VMARG );
    
    static FALCON_FUNC set_placement( VMARG );
    
    static FALCON_FUNC get_vadjustment( VMARG );
    
    static FALCON_FUNC set_vadjustment( VMARG );
    
    static FALCON_FUNC get_hadjustment( VMARG );
    
    static FALCON_FUNC set_hadjustment( VMARG );
    
#if GTK_CHECK_VERSION(2, 8, 0)
    
    static FALCON_FUNC get_hscrollbar( VMARG );
    
    static FALCON_FUNC get_vscrollbar( VMARG );
    
#endif //GTK_CHECK_VERSION(2, 8, 0)
    
#if GTK_CHECK_VERSION(2, 10, 0)
    
    static FALCON_FUNC unset_placement( VMARG );
    
#endif //GTK_CHECK_VERSION(2, 10, 0)
    
    static FALCON_FUNC set_shadow_type( VMARG );
    
    static FALCON_FUNC get_shadow_type( VMARG );
    
    static FALCON_FUNC signal_move_focus_out( VMARG );
    
    static void on_move_focus_out( GtkScrolledWindow *obj, gpointer _vm );
    
    static FALCON_FUNC signal_scroll_child( VMARG );
    
    static void on_scroll_child( GtkScrolledWindow *obj, gpointer _vm );
    
};

} //Gtk

} //Falcon


#endif //!GTK_SCROLLED_WINDOW_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;