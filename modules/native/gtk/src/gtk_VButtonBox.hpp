#ifndef GTK_VBUTTONBOX_HPP
#define GTK_VBUTTONBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::VButtonBox
 */
class VButtonBox
    :
    public Gtk::CoreGObject
{
public:

    VButtonBox( const Falcon::CoreClass*, const GtkVButtonBox* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    //static FALCON_FUNC get_spacing_default( VMARG );

    //static FALCON_FUNC get_layout_default( VMARG );

    //static FALCON_FUNC set_spacing_default( VMARG );

    //static FALCON_FUNC set_layout_default( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_VBUTTONBOX_HPP
