#ifndef GTK_SCALE_HPP
#define GTK_SCALE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Scale
 */
class Scale
    :
    public Gtk::CoreGObject
{
public:

    Scale( const Falcon::CoreClass*, const GtkScale* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC signal_format_value( VMARG );

    static gchar* on_format_value( GtkScale*, gdouble, gpointer );

    static FALCON_FUNC set_digits( VMARG );

    static FALCON_FUNC set_draw_value( VMARG );

    static FALCON_FUNC set_value_pos( VMARG );

    static FALCON_FUNC get_digits( VMARG );

    static FALCON_FUNC get_draw_value( VMARG );

    static FALCON_FUNC get_value_pos( VMARG );
#if 0 // todo
    static FALCON_FUNC get_layout( VMARG );

    static FALCON_FUNC get_layout_offsets( VMARG );

    static FALCON_FUNC add_mark( VMARG );

    static FALCON_FUNC clear_marks( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_SCALE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
