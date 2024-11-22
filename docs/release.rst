How to make a new release?
--------------------------

1. Ensure that the most recent version of Dr.Jit is checked out (including all
   submodules).

2. Check that the ``nanobind`` dependency version in ``pyroject.toml`` matches
   the version used in the submodule.

3. Ensure that the changelog is up to date in ``docs/changelog.rst``.

4. Verify that the CI is currently green all platforms.

5. Run the GHA "Build Python Wheels" with option "0". This effectively is a dry
   run of the wheel creation process.

6. If the action failed, fix whatever broke in the build process. If it succeded
   continue.

7. Update the version number in ``include/drjit/fwd.h``.

8. Add release number and date to ``docs/changelog.rst``.

9. Commit: ``git commit -am "vX.Y.Z release"``

10. Tag: ``git tag -a vX.Y.Z -m "vX.Y.Z release"``

11. Push: ``git push`` and ``git push --tags``

12. Run the GHA "Build Python Wheels" with option "1".

13. Check that the new version is available on
    `readthedocs <https://drjit.readthedocs.io/>`__.

14. Create a `release on GitHub <https://github.com/mitsuba-renderer/drjit/releases/new>`__
    from the tag created at step 10. The changelog can be copied from step 2.

15. Checkout the ``stable`` branch and run ``git pull --ff-only origin vX.Y.Z``
    and ``git push``
