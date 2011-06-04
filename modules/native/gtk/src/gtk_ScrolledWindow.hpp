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
    
//  static FALCON_FUNC get_hscrollbar( VMARG );
//  static FALCON_FUNC get_vscrollbar( VMARG );
//  static FALCON_FUNC unset_placement( VMARG );
//  static FALCON_FUNC set_shadow_type( VMARG );
//  static FALCON_FUNC get_shadow_type( VMARG );
            
};
        
} //Gtk
    
} //Falcon


#endif //!GTK_SCROLLED_WINDOW_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;