/**
 *  \file gtk_ToolShell.cpp
 */

#include "gtk_ToolShell.hpp"


#if GTK_MINOR_VERSION >= 14

namespace Falcon {
namespace Gtk {


/**
 *  \brief interface loader
 */
void ToolShell::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
#if GTK_MINOR_VERSION >= 20
    { "get_ellipsize_mode",     &ToolShell::get_ellipsize_mode },
#endif
    { "get_icon_size",          &ToolShell::get_icon_size },
    { "get_orientation",        &ToolShell::get_orientation },
    { "get_relief_style",       &ToolShell::get_relief_style },
    { "get_style",              &ToolShell::get_style },
#if GTK_MINOR_VERSION >= 20
    { "get_text_alignment",     &ToolShell::get_text_alignment },
    { "get_text_orientation",   &ToolShell::get_text_orientation },
#endif
    { "rebuild_menu",           &ToolShell::rebuild_menu },
#if GTK_MINOR_VERSION >= 20
    //{ "get_text_size_group",    &ToolShell::get_text_size_group },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


/*#
    @class GtkToolShell
    @brief Interface for containers containing GtkToolItem widgets

    The GtkToolShell interface allows container widgets to provide additional
    information when embedding GtkToolItem widgets.
 */


#if GTK_MINOR_VERSION >= 20
/*#
    @method get_ellipsize_mode GtkToolShell
    @brief Retrieves the current ellipsize mode for the tool shell.
    @return the current ellipsize mode of shell (PangoEllipsizeMode).

    Tool items must not call this function directly, but rely on
    gtk_tool_item_get_ellipsize_mode() instead.
 */
FALCON_FUNC ToolShell::get_ellipsize_mode( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_shell_get_ellipsize_mode( (GtkToolShell*)_obj ) );
}
#endif


/*#
    @method get_icon_size GtkToolShell
    @brief Retrieves the icon size for the tool shell.
    @return the current size for icons of shell (GtkIconSize).

    Tool items must not call this function directly, but rely on
    gtk_tool_item_get_icon_size() instead.
 */
FALCON_FUNC ToolShell::get_icon_size( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_shell_get_icon_size( (GtkToolShell*)_obj ) );
}


/*#
    @method get_orientation GtkToolShell
    @brief Retrieves the current orientation for the tool shell.
    @return the current orientation of shell (GtkOrientation).

    Tool items must not call this function directly, but rely on
    gtk_tool_item_get_orientation() instead.
 */
FALCON_FUNC ToolShell::get_orientation( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_shell_get_orientation( (GtkToolShell*)_obj ) );
}


/*#
    @method get_relief_style GtkToolShell
    @brief Returns the relief style of buttons on shell.
    @return The relief style of buttons on shell (GtkReliefStyle).

    Tool items must not call this function directly, but rely on
    gtk_tool_item_get_relief_style() instead.
 */
FALCON_FUNC ToolShell::get_relief_style( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_shell_get_relief_style( (GtkToolShell*)_obj ) );
}


/*#
    @method get_style GtkToolShell
    @brief Retrieves whether the tool shell has text, icons, or both.
    @return the current style of shell (GtkToolbarStyle).

    Tool items must not call this function directly, but rely on
    gtk_tool_item_get_style() instead.
 */
FALCON_FUNC ToolShell::get_style( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_shell_get_style( (GtkToolShell*)_obj ) );
}


#if GTK_MINOR_VERSION >= 20
/*#
    @method get_text_alignment GtkToolShell
    @brief Retrieves the current text alignment for the tool shell.
    @return the current text alignment of shell.

    Tool items must not call this function directly, but rely on
    gtk_tool_item_get_text_alignment() instead.
 */
FALCON_FUNC ToolShell::get_text_alignment( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_tool_shell_get_text_alignment( (GtkToolShell*)_obj ) );
}


/*#
    @method get_text_orientation GtkToolShell
    @brief Retrieves the current text orientation for the tool shell.
    @return the current text orientation of shell (GtkOrientation).

    Tool items must not call this function directly, but rely on
    gtk_tool_item_get_text_orientation() instead.
 */
FALCON_FUNC ToolShell::get_text_orientation( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_shell_get_text_orientation( (GtkToolShell*)_obj ) );
}
#endif // GTK_MINOR_VERSION >= 20


/*#
    @method rebuild_menu GtkToolShell
    @brief Calling this function signals the tool shell that the overflow menu item for tool items have changed.

    If there is an overflow menu and if it is visible when this function it
    called, the menu will be rebuilt.

    Tool items must not call this function directly, but rely on
    gtk_tool_item_rebuild_menu() instead.
 */
FALCON_FUNC ToolShell::rebuild_menu( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_shell_rebuild_menu( (GtkToolShell*)_obj );
}


#if GTK_MINOR_VERSION >= 20
//FALCON_FUNC ToolShell::get_text_size_group( VMARG );
#endif


} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 14
