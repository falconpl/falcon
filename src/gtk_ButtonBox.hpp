#ifndef GTK_BUTTONBOX_HPP
#define GTK_BUTTONBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ButtonBox
 */
class ButtonBox
    :
    public Gtk::CoreGObject
{
public:

    ButtonBox( const Falcon::CoreClass*, const GtkButtonBox* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    //static FALCON_FUNC get_layout( VMARG );

    static FALCON_FUNC get_child_size( VMARG );

    static FALCON_FUNC get_child_ipadding( VMARG );

    static FALCON_FUNC get_child_secondary( VMARG );

    //static FALCON_FUNC set_spacing( VMARG );

    //static FALCON_FUNC set_layout( VMARG );

    //static FALCON_FUNC set_child_size( VMARG );

    //static FALCON_FUNC set_child_ipadding( VMARG );

    static FALCON_FUNC set_child_secondary( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_BUTTONBOX_HPP
