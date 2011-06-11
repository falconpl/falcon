#ifndef GTK_FRAME_HPP
#define GTK_FRAME_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Frame
 */
class Frame
    :
    public Gtk::CoreGObject
{
public:

    Frame( const Falcon::CoreClass*, const GtkFrame* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_label( VMARG );

    static FALCON_FUNC set_label_widget( VMARG );

    static FALCON_FUNC set_label_align( VMARG );

    static FALCON_FUNC set_shadow_type( VMARG );

    static FALCON_FUNC get_label( VMARG );

    static FALCON_FUNC get_label_align( VMARG );

    static FALCON_FUNC get_label_widget( VMARG );

    static FALCON_FUNC get_shadow_type( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_FRAME_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
