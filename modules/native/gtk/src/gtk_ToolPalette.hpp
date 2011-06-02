#ifndef GTK_TOOLPALETTE_HPP
#define GTK_TOOLPALETTE_HPP

#include "modgtk.hpp"

#if GTK_CHECK_VERSION( 2, 20, 0 )

namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ToolPalette
 */
class ToolPalette
    :
    public Gtk::CoreGObject
{
public:

    ToolPalette( const Falcon::CoreClass*, const GtkToolPalette* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_set_scroll_adjustments( VMARG );

    static void on_set_scroll_adjustments( GtkToolPalette*, GtkAdjustment*, GtkAdjustment*, gpointer );

    static FALCON_FUNC get_exclusive( VMARG );

    static FALCON_FUNC set_exclusive( VMARG );

    static FALCON_FUNC get_expand( VMARG );

    static FALCON_FUNC set_expand( VMARG );

    static FALCON_FUNC get_group_position( VMARG );

    static FALCON_FUNC set_group_position( VMARG );

    static FALCON_FUNC get_icon_size( VMARG );

    static FALCON_FUNC set_icon_size( VMARG );

    static FALCON_FUNC unset_icon_size( VMARG );

    static FALCON_FUNC get_style( VMARG );

    static FALCON_FUNC set_style( VMARG );

    static FALCON_FUNC unset_style( VMARG );
#if 0 // todo
    static FALCON_FUNC add_drag_dest( VMARG );

    static FALCON_FUNC get_drag_item( VMARG );

    static FALCON_FUNC get_drag_target_group( VMARG );

    static FALCON_FUNC get_drag_target_item( VMARG );
#endif
    static FALCON_FUNC get_drop_group( VMARG );

    static FALCON_FUNC get_drop_item( VMARG );

    static FALCON_FUNC set_drag_source( VMARG );

    static FALCON_FUNC get_hadjustment( VMARG );

    static FALCON_FUNC get_vadjustment( VMARG );

};


} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 20, 0 )

#endif // !GTK_TOOLPALETTE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
