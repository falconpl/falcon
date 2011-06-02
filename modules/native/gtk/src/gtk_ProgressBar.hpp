#ifndef GTK_PROGRESSBAR_HPP
#define GTK_PROGRESSBAR_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ProgressBar
 */
class ProgressBar
    :
    public Gtk::CoreGObject
{
public:

    ProgressBar( const Falcon::CoreClass*, const GtkProgressBar* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC pulse( VMARG );

    static FALCON_FUNC set_text( VMARG );

    static FALCON_FUNC set_fraction( VMARG );

    static FALCON_FUNC set_pulse_step( VMARG );

    static FALCON_FUNC set_orientation( VMARG );

    static FALCON_FUNC set_ellipsize( VMARG );

    static FALCON_FUNC get_text( VMARG );

    static FALCON_FUNC get_fraction( VMARG );

    static FALCON_FUNC get_pulse_step( VMARG );

    static FALCON_FUNC get_orientation( VMARG );

    static FALCON_FUNC get_ellipsize( VMARG );

#if 0 // deprecated
    static FALCON_FUNC new_with_adjustment( VMARG );

    static FALCON_FUNC set_bar_style( VMARG );

    static FALCON_FUNC set_discrete_blocks( VMARG );

    static FALCON_FUNC set_activity_step( VMARG );

    static FALCON_FUNC set_activity_blocks( VMARG );

    static FALCON_FUNC update( VMARG );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_PROGRESSBAR_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
